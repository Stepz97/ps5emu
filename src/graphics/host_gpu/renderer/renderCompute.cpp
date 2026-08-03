#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorCache.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/pipeline/shaderSubgroup.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {
static uint64_t BufferDescriptorSize(const ShaderBufferResource& descriptor) {
	const uint64_t records = descriptor.NumRecords();
	const uint64_t stride  = descriptor.Stride();
	if (stride != 0 && records > UINT64_MAX / stride) {
		EXIT("compute buffer descriptor footprint overflow\n");
	}
	return stride == 0 ? records : records * stride;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const RenderCommandBuffer&    buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		if (!resource.written && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeImageClear(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (program.info.buffers.size() != 1 || resources.buffers.size() != 1 ||
	    !program.info.images.empty() || !program.info.samplers.empty() ||
	    !program.info.addresses.empty() || !resources.images.empty() ||
	    !resources.samplers.empty() || !resources.addresses.empty()) {
		return false;
	}
	const auto& resource   = program.info.buffers.front();
	const auto& raw        = resources.buffers.front();
	const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(raw);
	if (!resource.formatted || !resource.written || resource.read || resource.atomic ||
	    resource.scalar || resource.max_byte_extent != 16 || descriptor.Stride() != 16 ||
	    descriptor.Format() != Prospero::GpuEnumValue(Prospero::BufferFormat::k32_32_32_32UInt) ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    resource.packed_stride != descriptor.PackedStride() || raw.dword_count != 4 ||
	    program.user_data_base != 0 || resources.user_data.size() != 8) {
		return false;
	}
	for (uint32_t i = 0; i < raw.dword_count; i++) {
		if (raw.dwords[i] != resources.user_data[i]) {
			return false;
		}
	}
	const uint32_t clear = resources.user_data[4];
	if (resources.user_data[5] != clear || resources.user_data[6] != clear ||
	    resources.user_data[7] != clear) {
		return false;
	}
	const bool full_dispatch =
	    input.dispatch_thread_dimensions && input.threads_num[0] == 64 &&
	    input.threads_num[1] == 1 && input.threads_num[2] == 1 && group_x != 0 && group_y == 1 &&
	    group_z == 1 && input.dispatch_threads_num[0] == group_x &&
	    input.dispatch_threads_num[1] == 1 && input.dispatch_threads_num[2] == 1 &&
	    input.group_id[0] && !input.group_id[1] && !input.group_id[2] &&
	    input.thread_ids_num == 1 && input.wave_size == 32 && !input.tg_size_en && mode == 0x61u &&
	    group_x % input.threads_num[0] == 0 && descriptor.NumRecords() == group_x;
	const auto size = BufferDescriptorSize(descriptor);
	if (!full_dispatch || size == 0) {
		return false;
	}
	resolved_descriptor = descriptor;
	resolved_clear      = clear;
	resolved_size       = size;
	return true;
}

static bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input, CommandBuffer& command,
                                        uint32_t group_x, uint32_t group_y, uint32_t group_z,
                                        uint32_t mode) {
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeImageClear(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	auto& cache = command.GetContext().GetTextureCache();
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, RenderCommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid());
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (!ShaderAddressValid(sh_ctx.GetCs().cs_regs.data_addr)) {
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo    input_info {};
	std::span<const uint32_t> cs_shader;
	if (!ShaderCompileInfoCS(cs_regs, sh_regs, input_info, cs_shader)) {
		EXIT("ShaderCompileInfoCS failed for dispatch with CS shader 0x%016" PRIx64 "\n",
		     cs_regs.cs_regs.data_addr);
	}

	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	if (use_thread_dimensions) {
		input_info.dispatch_thread_dimensions = true;
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
	const bool     large_workgroup =
	    (input_info.threads_num[0] * input_info.threads_num[1] * input_info.threads_num[2] >= 512);
	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;

	// m42-writers (TEMPORARY, remove before commit): census of the dispatches that WRITE a
	// watched guest base (KYTY_M42_WRITERS=<hex,hex,...>), with the shader hash, the thread
	// group counts and the descriptor extents. Counting how many distinct dispatches cover a
	// buffer in the loading phase versus the menu phase says whether a producer stops being
	// emitted — the open question behind the quarter of the lighting output nobody writes.
	static const std::vector<uint64_t> m42_writer_bases = []() {
		std::vector<uint64_t> bases;
		if (const char* v = std::getenv("KYTY_M42_WRITERS"); v != nullptr) {
			const char* p = v;
			while (*p != '\0') {
				char*      end  = nullptr;
				const auto base = std::strtoull(p, &end, 16);
				if (end == p) {
					break;
				}
				if (base != 0) {
					bases.push_back(base);
				}
				p = (*end == ',') ? end + 1 : end;
			}
		}
		return bases;
	}();
	// Buffer side of the same census: a dispatch whose thread count comes from a counter the
	// GPU produced shows up here as a small written buffer. Logs the first dwords so a value
	// in the same range as the dispatch thread count is recognisable on sight.
	if (!m42_writer_bases.empty() && resources.buffers.size() == program.info.buffers.size()) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			if (!program.info.buffers[i].written) {
				continue;
			}
			const auto r = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			const uint64_t bb   = r.Base48();
			const uint64_t size = static_cast<uint64_t>(r.Stride()) * r.NumRecords();
			if (bb == 0 || size == 0 || size > 4096) {
				continue; // counters and small work-lists only
			}
			uint32_t words[4] = {0, 0, 0, 0};
			if (!LibKernel::Memory::TryReadBacking(bb, words, sizeof(words))) {
				continue;
			}
			const bool interesting = words[0] > 100000 && words[0] < 4000000;
			static Common::Mutex                          cnt_mutex;
			static std::unordered_map<uint64_t, uint32_t> cnt_seen;
			uint32_t seen = 0;
			{
				Common::LockGuard cnt_lock(cnt_mutex);
				seen = ++cnt_seen[bb ^ (cs_regs.cs_regs.data_addr << 1u)];
			}
			if (interesting && (seen <= 6 || (seen & 0x3ffu) == 0)) {
				LOGF("m42-counter: frame=%u shader=0x%016" PRIx64 " buf[%u] addr=0x%012" PRIx64
				     " size=%" PRIu64 " words=[%u %u %u %u] n=%u\n",
				     frame_num, cs_regs.cs_regs.data_addr, i, bb, size, words[0], words[1],
				     words[2], words[3], seen);
			}
		}
	}

	if (!m42_writer_bases.empty() && resources.images.size() == program.info.images.size()) {
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			if (!program.info.images[i].written) {
				continue;
			}
			const auto w = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			const auto wb = w.Base40();
			if (std::find(m42_writer_bases.begin(), m42_writer_bases.end(), wb) ==
			    m42_writer_bases.end()) {
				continue;
			}
			static Common::Mutex                            writer_mutex;
			static std::unordered_map<uint64_t, uint32_t>   writer_seen;
			const uint64_t key = cs_regs.cs_regs.data_addr ^ (wb << 1u) ^
			                     (static_cast<uint64_t>(i) << 48u);
			uint32_t seen_count = 0;
			{
				Common::LockGuard writer_lock(writer_mutex);
				seen_count = ++writer_seen[key];
			}
			if (seen_count <= 4 || (seen_count & 0xffu) == 0) {
				LOGF("m42-writers: frame=%u shader=0x%016" PRIx64 " img[%u] base=0x%010" PRIx64
				     " %ux%u fmt=%u groups=%ux%ux%u threads=%ux%ux%u n=%u\n",
				     frame_num, cs_regs.cs_regs.data_addr, i, wb,
				     static_cast<uint32_t>(w.Width5()) + 1u,
				     static_cast<uint32_t>(w.Height5()) + 1u, w.Format(), thread_group_x,
				     thread_group_y, thread_group_z, input_info.threads_num[0],
				     input_info.threads_num[1], input_info.threads_num[2], seen_count);
			}
		}
	}

	// scanout-probe: log the FIRST sighting of every guest address a compute shader can
	// write (buffers and storage images), to attribute which dispatch (if any) produces
	// the buffer the scan-out presents. Runs BEFORE the clear fast-paths so clear targets
	// are attributed too. Deduped; only active alongside --dump-scanout.
	static const bool probe_enabled = !Config::GetDumpScanOutPath().empty();
	if (probe_enabled && resources.buffers.size() == program.info.buffers.size() &&
	    resources.images.size() == program.info.images.size()) {
		static Common::Mutex                probe_mutex;
		static std::unordered_set<uint64_t> probe_seen;
		const auto probe_first_sight = [](uint64_t addr) {
			Common::LockGuard probe_lock(probe_mutex);
			return probe_seen.insert(addr).second;
		};
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			if (!program.info.buffers[i].written) {
				continue;
			}
			const auto r =
			    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			if (r.Base48() != 0 && probe_first_sight(r.Base48())) {
				LOGF("scanout-probe: cs buffer write base=0x%012" PRIx64 " size=0x%" PRIx64
				     " shader=0x%016" PRIx64 "\n",
				     r.Base48(), BufferDescriptorSize(r), cs_regs.cs_regs.data_addr);
			}
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			if (!program.info.images[i].written) {
				continue;
			}
			const auto r =
			    DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			if (r.Base40() != 0 && probe_first_sight(r.Base40())) {
				LOGF("scanout-probe: cs image write base=0x%010" PRIx64
				     " shader=0x%016" PRIx64 "\n",
				     r.Base40(), cs_regs.cs_regs.data_addr);
			}
		}
	}
	// m42-skip probe (TEMPORARY): dispatch whose SRT table was readable-but-empty smears
	// black with zeroed parameters; skip it so healthy dispatches keep their output.
	if (ShaderM42SkipDispatchRequested()) {
		static std::atomic<uint32_t> skip_count {0};
		const auto n = skip_count.fetch_add(1, std::memory_order_relaxed) + 1;
		if (n <= 16 || (n & 0x1ffu) == 0) {
			LOGF("m42-skip: dropping compute dispatch with empty SRT table shader=0x%016" PRIx64
			     " (n=%u)\n",
			     cs_regs.cs_regs.data_addr, n);
		}
		ResetBindings();
		return;
	}
	if (TryConsumeComputeMetaClear(input_info, buffer)) {
		ResetBindings();
		return;
	}
	if (TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
	                                thread_group_z, mode)) {
		ResetBindings();
		return;
	}
	const auto sampled_images = std::count_if(
	    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.kind == ShaderRecompiler::IR::ResourceKind::Image ||
		           image.kind == ShaderRecompiler::IR::ResourceKind::ImageUint;
	    });
	// m42-gbuf probe (TEMPORARY, remove before commit): the muro-42 "second key" candidate
	// says the lighting CS reads its G-buffer inputs black. For the first rounds of every
	// sampler-heavy CS, pull each sampled input image back to guest memory through the same
	// machinery the scan-out dump already uses, sample the backing, and log max/nonzero so
	// "inputs really black" vs "inputs have data, the key is elsewhere" is decided by
	// measurement, not inference. Gated by KYTY_M42_GBUF_PROBE=1.
	static const bool m42_gbuf_enabled = []() {
		const char* v = std::getenv("KYTY_M42_GBUF_PROBE");
		return v != nullptr && v[0] == '1';
	}();
	if (m42_gbuf_enabled && sampled_images >= 8 &&
	    resources.images.size() == program.info.images.size()) {
		static Common::Mutex gbuf_mutex;
		// (last_frame, count) per shader: rounds are spaced in time (one per shader every
		// 256 frames) instead of a fixed budget, so the menu phase gets probed too — a
		// flat cap burned all rounds during loading, whose descriptors are also real.
		static std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> gbuf_rounds;
		// Rounds are only consumed when the descriptors carry real bases: during the
		// loading phase the SRT tables are empty and every sampled descriptor decodes to
		// null, and burning the per-shader budget there would leave nothing for the menu
		// phase, the one this probe exists to measure.
		bool gbuf_any_base = false;
		for (uint32_t i = 0; i < program.info.images.size() && !gbuf_any_base; i++) {
			const auto& image = program.info.images[i];
			const bool  is_sampled =
			    image.kind == ShaderRecompiler::IR::ResourceKind::Image ||
			    image.kind == ShaderRecompiler::IR::ResourceKind::ImageUint;
			if (!is_sampled || image.written) {
				continue;
			}
			gbuf_any_base =
			    DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]).Base40() != 0;
		}
		if (!gbuf_any_base) {
			static std::atomic<uint32_t> gbuf_null_logs {0};
			if (gbuf_null_logs.fetch_add(1, std::memory_order_relaxed) < 8) {
				LOGF("m42-gbuf: frame=%u shader=0x%016" PRIx64
				     " all sampled descriptors null (empty SRT phase), round not consumed\n",
				     frame_num, cs_regs.cs_regs.data_addr);
			}
		}
		uint32_t gbuf_round = UINT32_MAX;
		if (gbuf_any_base) {
			Common::LockGuard gbuf_lock(gbuf_mutex);
			auto [it, inserted] =
			    gbuf_rounds.try_emplace(cs_regs.cs_regs.data_addr, std::pair {0u, 0u});
			if (inserted || frame_num >= it->second.first + 256u) {
				it->second.first = frame_num;
				gbuf_round       = it->second.second++;
			}
		}
		if (gbuf_round != UINT32_MAX) {
			for (uint32_t i = 0; i < program.info.images.size(); i++) {
				const auto& image      = program.info.images[i];
				const bool  is_sampled =
				    image.kind == ShaderRecompiler::IR::ResourceKind::Image ||
				    image.kind == ShaderRecompiler::IR::ResourceKind::ImageUint;
				if (image.written) {
					// Output side of the dispatch: knowing WHERE the lighting writes is
					// half of the downstream join (does the post chain / presenter read
					// this base?).
					const auto w =
					    DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
					LOGF("m42-gbuf-out: frame=%u round=%u shader=0x%016" PRIx64
					     " img[%u] base=0x%010" PRIx64 " fmt=%u %ux%u tile=%u\n",
					     frame_num, gbuf_round, cs_regs.cs_regs.data_addr, i, w.Base40(),
					     w.Format(), static_cast<uint32_t>(w.Width5()) + 1u,
					     static_cast<uint32_t>(w.Height5()) + 1u, w.TileMode());
					continue;
				}
				if (!is_sampled) {
					continue;
				}
				const auto r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
				const uint64_t base   = r.Base40();
				const uint32_t width  = static_cast<uint32_t>(r.Width5()) + 1u;
				const uint32_t height = static_cast<uint32_t>(r.Height5()) + 1u;
				if (base == 0) {
					LOGF("m42-gbuf: frame=%u round=%u shader=0x%016" PRIx64
					     " img[%u] null descriptor\n",
					     frame_num, gbuf_round, cs_regs.cs_regs.data_addr, i);
					continue;
				}
				const uint64_t est_size = std::clamp<uint64_t>(
				    static_cast<uint64_t>(width) * height * 4u, 0x1000u, 32u << 20u);
				// gpumod=1 already IS the payload ("a GPU producer wrote this texture"):
				// pulling the content would go through the readback download, which EXITs
				// on tiled formats it cannot detile (fmt=5 tile=24, seen live killing the
				// first probe boot). Bytes are only sampled when the tracker holds no
				// GPU-side truth, where the raw backing read is fault-free and honest.
				const bool gpu_mod =
				    LibKernel::Memory::IsGpuBufferRegionModified(base, est_size);
				std::array<uint8_t, 4096> gbuf_chunk {};
				uint32_t                  max_byte = 0;
				uint64_t                  nonzero  = 0;
				uint64_t                  total    = 0;
				if (!gpu_mod) {
					for (uint32_t c = 0; c < 16; c++) {
						const uint64_t offset = ((est_size * c) / 16u) & ~0xfffull;
						if (!LibKernel::Memory::TryReadBacking(base + offset, gbuf_chunk.data(),
						                                       gbuf_chunk.size())) {
							continue;
						}
						for (const auto b: gbuf_chunk) {
							max_byte = std::max<uint32_t>(max_byte, b);
							nonzero += (b != 0 ? 1u : 0u);
						}
						total += gbuf_chunk.size();
					}
				}
				LOGF("m42-gbuf: frame=%u round=%u shader=0x%016" PRIx64 " img[%u] base=0x%010" PRIx64
				     " fmt=%u %ux%u tile=%u gpumod=%d max=%u nonzero=%.1f%% bytes=%" PRIu64 "\n",
				     frame_num, gbuf_round, cs_regs.cs_regs.data_addr, i, base, r.Format(), width,
				     height, r.TileMode(), gpu_mod ? 1 : 0, max_byte,
				     total != 0 ? 100.0 * static_cast<double>(nonzero) / static_cast<double>(total)
				                : 0.0,
				     total);
			}
		}
	}
	// m42-trace probe (TEMPORARY, remove before commit): follow specific guest bases
	// through the compute side of the frame graph. KYTY_M42_TRACE_BASES=hex,hex,... logs
	// every dispatch that samples or writes a traced base (no minimum image count, unlike
	// m42-gbuf), and for LINEAR traced images also pulls + samples the real content — the
	// detile EXIT only bites tiled formats, and the exposure texture (48x27x64, tile=0)
	// was already pulled safely by the first probe generation.
	static const std::vector<uint64_t> m42_trace_bases = []() {
		std::vector<uint64_t> bases;
		if (const char* v = std::getenv("KYTY_M42_TRACE_BASES"); v != nullptr) {
			const char* p = v;
			while (*p != '\0') {
				char*      end  = nullptr;
				const auto base = std::strtoull(p, &end, 16);
				if (end == p) {
					break;
				}
				if (base != 0) {
					bases.push_back(base);
				}
				p = (*end == ',') ? end + 1 : end;
			}
		}
		return bases;
	}();
	if (!m42_trace_bases.empty() && resources.images.size() == program.info.images.size() &&
	    resources.buffers.size() == program.info.buffers.size()) {
		static Common::Mutex                          trace_mutex;
		static std::unordered_map<uint64_t, uint32_t> trace_last_frame;
		bool trace_hit = false;
		for (uint32_t i = 0; i < program.info.images.size() && !trace_hit; i++) {
			const auto base =
			    DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]).Base40();
			trace_hit = base != 0 && std::find(m42_trace_bases.begin(), m42_trace_bases.end(),
			                                   base) != m42_trace_bases.end();
		}
		if (trace_hit) {
			bool trace_due = false;
			{
				Common::LockGuard trace_lock(trace_mutex);
				auto& last = trace_last_frame[cs_regs.cs_regs.data_addr];
				if (last == 0 || frame_num >= last + 128u) {
					last      = std::max(frame_num, 1u);
					trace_due = true;
				}
			}
			if (trace_due) {
				LOGF("m42-trace: frame=%u shader=0x%016" PRIx64 " groups=%ux%ux%u textures=%zu "
				     "buffers=%zu\n",
				     frame_num, cs_regs.cs_regs.data_addr, thread_group_x, thread_group_y,
				     thread_group_z, program.info.images.size(), program.info.buffers.size());
				for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
					const auto r = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
					LOGF("m42-trace:   buf[%u] usage=%s addr=0x%012" PRIx64 " stride=%u records=%u\n",
					     i, program.info.buffers[i].written ? "read-write" : "read-only", r.Base48(),
					     r.Stride(), r.NumRecords());
					const uint64_t bufsize =
					    static_cast<uint64_t>(r.Stride()) * r.NumRecords();
					if (r.Base48() != 0 && bufsize > 0 && bufsize <= 256) {
						std::array<uint8_t, 256> raw {};
						if (LibKernel::Memory::TryReadBacking(r.Base48(), raw.data(), bufsize)) {
							char hex[3 * 32 + 1] = {};
							const auto n = std::min<uint64_t>(bufsize, 32);
							for (uint64_t b = 0; b < n; b++) {
								std::snprintf(hex + b * 3, 4, "%02x ", raw[b]);
							}
							float f[8] = {};
							std::memcpy(f, raw.data(), std::min<uint64_t>(bufsize, sizeof(f)));
							LOGF("m42-trace:     content hex=[%s] f32=[%g %g %g %g %g %g %g %g]\n",
							     hex, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
						}
					}
				}
				for (uint32_t i = 0; i < program.info.images.size(); i++) {
					const auto r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
					const uint64_t base   = r.Base40();
					const uint32_t width  = static_cast<uint32_t>(r.Width5()) + 1u;
					const uint32_t height = static_cast<uint32_t>(r.Height5()) + 1u;
					const uint32_t depth  = static_cast<uint32_t>(r.Depth()) + 1u;
					uint32_t max_byte = 0;
					uint64_t nonzero  = 0;
					uint64_t total    = 0;
					if (base != 0 && r.TileMode() == 0) {
						const uint64_t est = std::clamp<uint64_t>(
						    static_cast<uint64_t>(width) * height * std::max(depth, 1u) * 4u, 0x1000u,
						    32u << 20u);
						(void)LibKernel::Memory::SynchronizeGpuImageToMemory(base, est);
						std::array<uint8_t, 4096> chunk {};
						for (uint32_t c = 0; c < 16; c++) {
							const uint64_t offset = ((est * c) / 16u) & ~0xfffull;
							if (!LibKernel::Memory::TryReadBacking(base + offset, chunk.data(),
							                                       chunk.size())) {
								continue;
							}
							for (const auto b: chunk) {
								max_byte = std::max<uint32_t>(max_byte, b);
								nonzero += (b != 0 ? 1u : 0u);
							}
							total += chunk.size();
						}
					}
					LOGF("m42-trace:   img[%u] usage=%s sampled=%d base=0x%010" PRIx64
					     " fmt=%u %ux%ux%u tile=%u max=%u nonzero=%.1f%% bytes=%" PRIu64 "\n",
					     i, program.info.images[i].written ? "read-write" : "read-only",
					     (program.info.images[i].kind == ShaderRecompiler::IR::ResourceKind::Image ||
					      program.info.images[i].kind == ShaderRecompiler::IR::ResourceKind::ImageUint)
					         ? 1
					         : 0,
					     base, r.Format(), width, height, depth, r.TileMode(), max_byte,
					     total != 0
					         ? 100.0 * static_cast<double>(nonzero) / static_cast<double>(total)
					         : 0.0,
					     total);
				}
			}
		}
	}
	const bool                   has_sampler = !program.info.samplers.empty();
	static std::atomic<uint32_t> dispatch_log_count {0};
	if ((large_workgroup || has_sampler) &&
	    dispatch_log_count.fetch_add(1, std::memory_order_relaxed) < 512) {
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y,
		     thread_group_z, mode, input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(), program.bindings.push_constant_size);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.Format());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     (image.kind == ShaderRecompiler::IR::ResourceKind::Image ||
			      image.kind == ShaderRecompiler::IR::ResourceKind::ImageUint)
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), r.Format(),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     std::max<uint32_t>(static_cast<uint32_t>(r.LastLevel()),
			                        static_cast<uint32_t>(r.MaxMip())) +
			         1u,
			     r.TileMode());
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(resources.samplers[i]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	if (use_thread_dimensions) {
		auto groups_from_threads = [](uint32_t threads, uint32_t group_size) {
			return (threads == 0
			            ? 0u
			            : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
		};

		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = groups_from_threads(thread_group_x, cs_regs.cs_regs.num_thread_x);
		thread_group_y       = groups_from_threads(thread_group_y, cs_regs.cs_regs.num_thread_y);
		thread_group_z       = groups_from_threads(thread_group_z, cs_regs.cs_regs.num_thread_z);

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	buffer.EndRendering();
	auto& pipeline =
	    m_context.GetPipelineCache().CreateComputePipeline(input_info, sh_ctx.GetCs(), cs_shader);
	auto bindings = PrepareBindings(buffer, input_info.stage, vk::ShaderStageFlagBits::eCompute,
	                                DescriptorCache::Stage::Compute);
	RebindBuffers(buffer, bindings);
	RebindImages(buffer, bindings);

	auto vk_buffer = buffer.Handle();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline.pipeline_layout, bindings);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);

	bool has_storage_writes = HasShaderBufferWrites(input_info.stage);
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       (image.kind == ShaderRecompiler::IR::ResourceKind::StorageImage ||
		                        image.kind == ShaderRecompiler::IR::ResourceKind::StorageImageUint);
	                }) ||
	    has_storage_writes;
	if (has_storage_writes) {
		ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	ResetBindings();
}

} // namespace Libs::Graphics
