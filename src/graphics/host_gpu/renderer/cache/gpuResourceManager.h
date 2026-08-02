#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_

#include "common/abi.h"
#include "common/common.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/resourceMutex.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include <cstdint>
#include <cstdio>
#include <shared_mutex>

namespace Libs::Graphics {

class CommandScheduler;
class Gpu;

class GpuResourceManager {
public:
	GpuResourceManager(GraphicContext& graphics, CommandScheduler& scheduler);
	~GpuResourceManager();
	KYTY_CLASS_NO_COPY(GpuResourceManager);

	[[nodiscard]] BufferCache&  GetBufferCache() { return m_buffer_cache; }
	[[nodiscard]] TextureCache& GetTextureCache() { return m_texture_cache; }
	void                        SetGpu(Gpu* gpu) noexcept { m_gpu = gpu; }

	[[nodiscard]] bool HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept;
	void DumpWatchedRanges(std::FILE* out) const noexcept { m_page_manager.DumpWatchedRanges(out); }
	[[nodiscard]] bool InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Pull an image that lives at this address back into guest memory. The inverse of
	// InvalidateMemory, and the only way instrumentation outside the renderer can look at
	// what was actually drawn: the guest scan-out buffer is never written otherwise.
	[[nodiscard]] bool SynchronizeImageToMemory(uint64_t vaddr, uint64_t size);
	// Pull GPU-modified buffer bytes covering this range back into guest memory. The buffer
	// twin of SynchronizeImageToMemory: GPU-produced tables (e.g. SRT chains written by an
	// earlier dispatch) live in cached buffers the CPU backing has never seen, and any host
	// read that bypasses the fault path (shader-walk reads) sees stale zeros otherwise.
	// No-op unless the containing tracker pages are GPU-modified.
	[[nodiscard]] bool SynchronizeBufferToMemory(uint64_t vaddr, uint64_t size);
	// True when any tracker page covering the range is GPU-modified (diagnostics).
	[[nodiscard]] bool IsBufferRegionGpuModified(uint64_t vaddr, uint64_t size);
	// True when any tracker page covering the range is CPU-modified (guest wrote it).
	[[nodiscard]] bool IsBufferRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsMapped(uint64_t vaddr, uint64_t size) const noexcept;
	void               MapMemory(uint64_t vaddr, uint64_t size);
	void               UnmapMemory(uint64_t vaddr, uint64_t size);
	void               RunGarbageCollector();

private:
	PageManager               m_page_manager;
	ResourceMutex             m_resource_mutex;
	BufferCache               m_buffer_cache;
	TextureCache              m_texture_cache;
	mutable std::shared_mutex m_mapped_ranges_mutex;
	RangeSet                  m_mapped_ranges;
	Gpu*                      m_gpu = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPURESOURCEMANAGER_H_
