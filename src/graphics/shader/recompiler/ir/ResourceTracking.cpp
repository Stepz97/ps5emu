#include "graphics/shader/recompiler/ir/ResourceTracking.h"

#include "graphics/shader/recompiler/ir/ScalarProvenance.h"

#include <algorithm>
#include <fmt/format.h>
#include <functional>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

const char* StageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vertex";
		case ShaderType::Pixel: return "pixel";
		case ShaderType::Fetch: return "fetch";
		case ShaderType::Compute: return "compute";
		default: return "unknown";
	}
}

bool IsAtomic(Opcode op) {
	switch (op) {
		case Opcode::AtomicSwapU32:
		case Opcode::AtomicAddU32:
		case Opcode::AtomicSubU32:
		case Opcode::AtomicSMinI32:
		case Opcode::AtomicUMinU32:
		case Opcode::AtomicSMaxI32:
		case Opcode::AtomicUMaxU32:
		case Opcode::AtomicAndU32:
		case Opcode::AtomicOrU32:
		case Opcode::AtomicXorU32: return true;
		default: return false;
	}
}

bool IsWrite(Opcode op) {
	switch (op) {
		case Opcode::BufferStoreByte:
		case Opcode::BufferStoreShort:
		case Opcode::BufferStoreDword:
		case Opcode::ImageStore: return true;
		default: return IsAtomic(op);
	}
}

bool NeedsSampler(Opcode op) {
	return op == Opcode::ImageSample || op == Opcode::ImageGather4 || op == Opcode::ImageGetLod;
}

bool IsDepthCompare(const Instruction& inst) {
	return (inst.memory.image_sample_flags & Decoder::ImageSampleFlagCompare) != 0;
}

ImageMipMode MipMode(const Instruction& inst) {
	const bool storage = inst.memory.kind == ResourceKind::StorageImage ||
	                     inst.memory.kind == ResourceKind::StorageImageUint;
	return storage && inst.memory.image_has_mip ? ImageMipMode::DynamicStorage : ImageMipMode::None;
}

bool IsBuffer(const Instruction& inst) {
	return inst.memory.kind == ResourceKind::Buffer ||
	       (inst.memory.kind == ResourceKind::ScalarBuffer && inst.op == Opcode::SBufferLoadDword);
}

bool IsImage(const Instruction& inst) {
	return inst.memory.kind == ResourceKind::Image || inst.memory.kind == ResourceKind::ImageUint ||
	       inst.memory.kind == ResourceKind::StorageImage ||
	       inst.memory.kind == ResourceKind::StorageImageUint;
}

bool IsAddress(const Instruction& inst) {
	return inst.op == Opcode::SLoadDword || inst.memory.kind == ResourceKind::Flat ||
	       inst.memory.kind == ResourceKind::Global || inst.memory.kind == ResourceKind::Scratch;
}

uint32_t ByteExtent(const Instruction& inst) {
	const auto bytes = std::max((inst.memory.data_bits + 7u) / 8u, 1u);
	const auto count = std::max(inst.memory.data_dwords, 1u);
	const auto end =
	    static_cast<uint64_t>(inst.memory.offset) + static_cast<uint64_t>(bytes) * count;
	return end > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(end);
}

bool ContainsUnknown(const ScalarProvenance& provenance, uint32_t id, std::vector<uint8_t>& visited,
	                 std::vector<uint32_t>& path) {
	path.push_back(id);
	if (id <= ScalarProvenance::Unknown || id >= provenance.values.size()) {
		return true;
	}
	if (visited[id] != 0) {
		path.pop_back();
		return false;
	}
	visited[id]       = 1;
	const auto& value = provenance.values[id];
	if (value.op == ScalarValueOp::Phi) {
		for (const auto arg: value.phi_args) {
			if (ContainsUnknown(provenance, arg, visited, path)) {
				return true;
			}
		}
		path.pop_back();
		return false;
	}
	const auto args = ScalarValueArgCount(value.op);
	for (uint32_t i = 0; i < args; i++) {
		if (ContainsUnknown(provenance, value.args[i], visited, path)) {
			return true;
		}
	}
	path.pop_back();
	return false;
}

bool IsLoopInvariantValue(const ScalarProvenance& provenance, uint32_t id,
	                      std::vector<uint8_t>& visiting) {
	if (id <= ScalarProvenance::Unknown || id >= provenance.values.size()) {
		return false;
	}
	if (visiting[id] != 0) {
		return true;
	}
	visiting[id]      = 1;
	const auto& value = provenance.values[id];
	if (value.op == ScalarValueOp::Phi) {
		uint32_t invariant = ScalarProvenance::Undefined;
		for (const auto arg: value.phi_args) {
			if (arg == id) {
				continue;
			}
			if (invariant == ScalarProvenance::Undefined) {
				invariant = arg;
			} else if (arg != invariant) {
				return false;
			}
		}
		return invariant != ScalarProvenance::Undefined &&
		       IsLoopInvariantValue(provenance, invariant, visiting);
	}
	for (uint32_t i = 0; i < ScalarValueArgCount(value.op); i++) {
		if (!IsLoopInvariantValue(provenance, value.args[i], visiting)) {
			return false;
		}
	}
	return true;
}

bool IsLoopInvariantDescriptor(const ScalarProvenance& provenance,
                               const DescriptorValue&  descriptor) {
	std::vector<uint8_t> visiting(provenance.values.size());
	for (uint32_t i = 0; i < descriptor.dword_count; i++) {
		if (!IsLoopInvariantValue(provenance, descriptor.dwords[i], visiting)) {
			return false;
		}
	}
	return true;
}

class Tracker {
public:
	explicit Tracker(Program& program): m_program(program) {}

	bool Run(std::string* error) {
		if (m_program.resource_tracking_complete) {
			return Fail(0, error, "resources already tracked");
		}
		if (!m_program.srt_plan_complete) {
			return Fail(0, error, "SRT plan is not ready");
		}
		if (!m_program.srt_patching_complete) {
			return Fail(0, error, "SRT reads were not patched");
		}
		// Keep scanning after a failed instruction so one report lists every unresolved
		// descriptor source instead of only the first.
		std::string all_failures;
		for (auto& block: m_program.blocks) {
			for (auto& inst: block.instructions) {
				std::string local;
				if (!Collect(inst, &local)) {
					if (!all_failures.empty()) {
						all_failures += '\n';
					}
					all_failures += local;
				}
			}
		}
		if (!all_failures.empty()) {
			if (error != nullptr) {
				*error = all_failures;
			}
			return false;
		}
		LinkImageAliases();
		m_program.info = std::move(m_info);
		for (const auto& patch: m_patches) {
			auto& inst = patch.inst.get();
			if (patch.degrade_dynamic_base) {
				inst.memory.dynamic_base        = true;
				inst.memory.dynamic_base_vsharp = patch.degrade_vsharp;
				inst.memory.dynamic_base_reg    = patch.degrade_base_reg;
				if (patch.degrade_vsharp) {
					inst.op = Opcode::SLoadDword;
				}
			}
			inst.memory.resource        = patch.resource;
			inst.memory.sampler         = patch.sampler;
			inst.memory.resource_source = ScalarProvenance::Undefined;
			inst.memory.sampler_source  = ScalarProvenance::Undefined;
		}
		m_program.resource_tracking_complete = true;
		return true;
	}

private:
	struct Patch {
		std::reference_wrapper<Instruction> inst;
		uint32_t                            resource         = 0;
		uint32_t                            sampler          = 0;
		// Deferred dynamic-base degradation, applied with the rest of the patch so a later
		// collection failure leaves the program untouched.
		bool     degrade_dynamic_base = false;
		bool     degrade_vsharp       = false;
		uint32_t degrade_base_reg     = 0;
	};

	bool Fail(uint32_t pc, std::string* error, const std::string& reason) const {
		if (error != nullptr) {
			*error = fmt::format("shader resource tracking: hash=0x{:016x} stage={} pc=0x{:08x} {}",
			                     m_program.shader_hash, StageName(m_program.stage), pc, reason);
		}
		return false;
	}

	bool ValidateSource(const Instruction& inst, uint32_t source, uint32_t dwords,
	                    std::string* error, bool* contains_unknown = nullptr) const {
		const auto  pc      = inst.pc;
		const auto  context = fmt::format("op={} kind={} component={}/{}",
		                                  static_cast<uint32_t>(inst.op),
		                                  static_cast<uint32_t>(inst.memory.kind),
		                                  inst.memory.component_index, inst.memory.component_count);
		const auto* descriptor = GetDescriptorSource(m_program, source);
		if (descriptor == nullptr || descriptor->dword_count != dwords) {
			return Fail(pc, error,
			            fmt::format("descriptor source {} is missing or has wrong width ({})",
			                        source, context));
		}
		std::vector<uint8_t> visited(m_program.provenance.values.size());
		for (uint32_t i = 0; i < descriptor->dword_count; i++) {
			std::vector<uint32_t> path;
			if (ContainsUnknown(m_program.provenance, descriptor->dwords[i], visited, path)) {
				if (contains_unknown != nullptr) {
					*contains_unknown = true;
				}
				const auto  value = descriptor->dwords[i];
				std::string chain;
				for (const auto id: path) {
					const auto op = id < m_program.provenance.values.size()
					                    ? static_cast<uint32_t>(m_program.provenance.values[id].op)
					                    : UINT32_MAX;
					chain += fmt::format("{}{}:{}({})", chain.empty() ? "" : " -> ", id, op,
					                     ScalarValueToString(m_program.provenance, id));
				}
				return Fail(
				    pc, error,
				    fmt::format(
				        "descriptor source {} dword {} contains an unknown value {} ({}) [{}] path {}",
				        source, i, value, ScalarValueToString(m_program.provenance, value), context,
				        chain));
			}
		}
		const auto dynamic =
		    std::find(m_program.srt.dynamic_sources.begin(), m_program.srt.dynamic_sources.end(),
		              source) != m_program.srt.dynamic_sources.end();
		if (dynamic && !IsLoopInvariantDescriptor(m_program.provenance, *descriptor)) {
			std::string detail;
			for (uint32_t i = 0; i < descriptor->dword_count; i++) {
				const auto id = descriptor->dwords[i];
				detail +=
				    fmt::format(" d{}={}({})", i, id,
				                id < m_program.provenance.values.size()
				                    ? static_cast<uint32_t>(m_program.provenance.values[id].op)
				                    : UINT32_MAX);
				if (id < m_program.provenance.values.size()) {
					for (const auto arg: m_program.provenance.values[id].phi_args) {
						detail += fmt::format(
						    "/{}({})", arg,
						    arg < m_program.provenance.values.size()
						        ? static_cast<uint32_t>(m_program.provenance.values[arg].op)
						        : UINT32_MAX);
					}
				}
			}
			return Fail(pc, error,
			            fmt::format("descriptor source {} requires unsupported GPU selection{}",
			                        source, detail));
		}
		if (!DescriptorSourceResolved(m_program, source) && !dynamic) {
			return Fail(pc, error, fmt::format("descriptor source {} is unresolved", source));
		}
		return true;
	}

	uint32_t AddBuffer(const Instruction& inst) {
		for (uint32_t i = 0; i < m_info.buffers.size(); i++) {
			if (m_info.buffers[i].source == inst.memory.resource_source) {
				Merge(&m_info.buffers[i], inst);
				return i;
			}
		}
		if (m_info.buffers.size() >= ShaderInfo::MaxBuffers) {
			return UINT32_MAX;
		}
		BufferResource resource;
		resource.source       = inst.memory.resource_source;
		resource.first_use_pc = inst.pc;
		Merge(&resource, inst);
		m_info.buffers.push_back(resource);
		return static_cast<uint32_t>(m_info.buffers.size() - 1);
	}

	void Merge(BufferResource* resource, const Instruction& inst) const {
		resource->first_use_pc    = std::min(resource->first_use_pc, inst.pc);
		resource->max_byte_extent = std::max(resource->max_byte_extent, ByteExtent(inst));
		resource->read            = resource->read || !IsWrite(inst.op) || IsAtomic(inst.op);
		resource->written         = resource->written || IsWrite(inst.op);
		resource->atomic          = resource->atomic || IsAtomic(inst.op);
		resource->formatted       = resource->formatted || inst.memory.formatted;
		resource->scalar = resource->scalar || inst.memory.kind == ResourceKind::ScalarBuffer;
	}

	void LinkImageAliases() {
		for (auto& buffer: m_info.buffers) {
			const auto* buffer_source = GetDescriptorSource(m_program, buffer.source);
			if (buffer_source == nullptr || buffer_source->dword_count != 4) {
				continue;
			}
			for (uint32_t i = 0; i < m_info.images.size(); i++) {
				const auto* image_source = GetDescriptorSource(m_program, m_info.images[i].source);
				if (image_source != nullptr && image_source->dword_count == 8 &&
				    std::equal(buffer_source->dwords.begin(), buffer_source->dwords.begin() + 4,
				               image_source->dwords.begin())) {
					buffer.image_alias = i;
					break;
				}
			}
		}
	}

	uint32_t AddImage(const Instruction& inst) {
		for (uint32_t i = 0; i < m_info.images.size(); i++) {
			auto& image = m_info.images[i];
			if (image.source == inst.memory.resource_source && image.kind == inst.memory.kind &&
			    image.dimension == inst.memory.image_dimension && image.mip_mode == MipMode(inst) &&
			    image.depth_compare == IsDepthCompare(inst)) {
				Merge(&image, inst);
				return i;
			}
		}
		if (m_info.images.size() >= ShaderInfo::MaxImages) {
			return UINT32_MAX;
		}
		ImageResource image;
		image.source        = inst.memory.resource_source;
		image.first_use_pc  = inst.pc;
		image.kind          = inst.memory.kind;
		image.dimension     = inst.memory.image_dimension;
		image.mip_mode      = MipMode(inst);
		image.depth_compare = IsDepthCompare(inst);
		Merge(&image, inst);
		m_info.images.push_back(image);
		return static_cast<uint32_t>(m_info.images.size() - 1);
	}

	uint32_t AddAddress(const Instruction& inst, bool dynamic = false) {
		auto immediate = static_cast<int32_t>(inst.memory.offset);
		if (inst.memory.kind == ResourceKind::ScalarBuffer) {
			immediate = static_cast<int32_t>(static_cast<uint32_t>(immediate) & ~3u);
		}
		const auto min_offset =
		    inst.memory.resource_source == ScalarProvenance::Unknown || dynamic
		        ? 0
		        : std::min(immediate, 0);
		for (uint32_t i = 0; i < m_info.addresses.size(); i++) {
			auto& address = m_info.addresses[i];
			if (address.source == inst.memory.resource_source && address.kind == inst.memory.kind &&
			    address.dynamic_base == dynamic) {
				address.first_use_pc = std::min(address.first_use_pc, inst.pc);
				address.min_offset   = std::min(address.min_offset, min_offset);
				address.read         = address.read || !IsWrite(inst.op) || IsAtomic(inst.op);
				address.written      = address.written || IsWrite(inst.op);
				address.atomic       = address.atomic || IsAtomic(inst.op);
				return i;
			}
		}
		if (m_info.addresses.size() >= ShaderInfo::MaxAddresses) {
			return UINT32_MAX;
		}
		AddressResource address {inst.memory.resource_source, inst.pc, inst.memory.kind,
		                         min_offset};
		address.read         = !IsWrite(inst.op) || IsAtomic(inst.op);
		address.written      = IsWrite(inst.op);
		address.atomic       = IsAtomic(inst.op);
		address.dynamic_base = dynamic;
		m_info.addresses.push_back(address);
		return static_cast<uint32_t>(m_info.addresses.size() - 1);
	}

	// Lowers a scalar load whose descriptor chain contains an Unknown to the dynamic-base
	// path: the emitted shader computes the full guest address from the live SGPRs that
	// already hold it, and translates it against a window anchored per dispatch by
	// best-effort evaluation of this same source chain. The instruction itself is only
	// rewritten when the whole collection succeeds.
	bool DegradeToDynamicBase(Instruction& inst, bool vsharp, std::string* error) {
		const auto resource = AddAddress(inst, true);
		if (resource == UINT32_MAX) {
			return Fail(inst.pc, error, "address resource limit exceeded");
		}
		Patch patch {std::ref(inst), resource, 0};
		patch.degrade_dynamic_base = true;
		patch.degrade_vsharp       = vsharp;
		patch.degrade_base_reg     = vsharp ? inst.memory.resource * 4u : inst.memory.resource;
		m_patches.push_back(patch);
		return true;
	}

	void Merge(ImageResource* resource, const Instruction& inst) const {
		resource->first_use_pc = std::min(resource->first_use_pc, inst.pc);
		resource->read         = resource->read || !IsWrite(inst.op) || IsAtomic(inst.op);
		resource->written      = resource->written || IsWrite(inst.op);
		resource->atomic       = resource->atomic || IsAtomic(inst.op);
	}

	uint32_t AddSampler(const Instruction& inst) {
		for (uint32_t i = 0; i < m_info.samplers.size(); i++) {
			if (m_info.samplers[i].source == inst.memory.sampler_source) {
				m_info.samplers[i].first_use_pc =
				    std::min(m_info.samplers[i].first_use_pc, inst.pc);
				return i;
			}
		}
		if (m_info.samplers.size() >= ShaderInfo::MaxSamplers) {
			return UINT32_MAX;
		}
		m_info.samplers.push_back({inst.memory.sampler_source, inst.pc});
		return static_cast<uint32_t>(m_info.samplers.size() - 1);
	}

	bool AddSampledPair(uint32_t image, uint32_t sampler, uint32_t pc, std::string* error) {
		for (auto& pair: m_info.sampled_pairs) {
			if (pair.image == image && pair.sampler == sampler) {
				pair.first_use_pc = std::min(pair.first_use_pc, pc);
				return true;
			}
		}
		if (m_info.sampled_pairs.size() >= ShaderInfo::MaxSampledPairs) {
			return Fail(pc, error, "sampled image/sampler pair limit exceeded");
		}
		m_info.sampled_pairs.push_back({image, sampler, pc});
		return true;
	}

	bool Collect(Instruction& inst, std::string* error) {
		if (IsAddress(inst)) {
			const bool unbased = inst.memory.resource_source == ScalarProvenance::Unknown;
			if (!unbased) {
				bool contains_unknown = false;
				if (!ValidateSource(inst, inst.memory.resource_source, 2, error,
				                    &contains_unknown)) {
					if (contains_unknown && inst.op == Opcode::SLoadDword) {
						return DegradeToDynamicBase(inst, false, error);
					}
					return false;
				}
			} else if (inst.memory.kind != ResourceKind::Flat &&
			           inst.memory.kind != ResourceKind::Global &&
			           inst.memory.kind != ResourceKind::Scratch) {
				return Fail(inst.pc, error, "scalar memory base is unresolved");
			}
			const auto resource = AddAddress(inst);
			if (resource == UINT32_MAX) {
				return Fail(inst.pc, error, "address resource limit exceeded");
			}
			m_patches.push_back({std::ref(inst), resource, 0});
			return true;
		}
		if (!IsBuffer(inst) && !IsImage(inst)) {
			return true;
		}
		{
			bool contains_unknown = false;
			if (!ValidateSource(inst, inst.memory.resource_source, IsBuffer(inst) ? 4u : 8u,
			                    error, &contains_unknown)) {
				if (contains_unknown && inst.op == Opcode::SBufferLoadDword &&
				    inst.memory.kind == ResourceKind::ScalarBuffer) {
					return DegradeToDynamicBase(inst, true, error);
				}
				return false;
			}
		}
		const auto resource = IsBuffer(inst) ? AddBuffer(inst) : AddImage(inst);
		if (resource == UINT32_MAX) {
			return Fail(inst.pc, error,
			            IsBuffer(inst) ? "buffer resource limit exceeded"
			                           : "image resource limit exceeded");
		}
		uint32_t sampler = 0;
		if (NeedsSampler(inst.op)) {
			if (!ValidateSource(inst, inst.memory.sampler_source, 4, error)) {
				return false;
			}
			sampler = AddSampler(inst);
			if (sampler == UINT32_MAX) {
				return Fail(inst.pc, error, "sampler resource limit exceeded");
			}
			if (!AddSampledPair(resource, sampler, inst.pc, error)) {
				return false;
			}
		}
		m_patches.push_back({std::ref(inst), resource, sampler});
		return true;
	}

	Program&           m_program;
	ShaderInfo         m_info;
	std::vector<Patch> m_patches;
};

} // namespace

bool TrackResources(Program& program, std::string* error) {
	return Tracker(program).Run(error);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
