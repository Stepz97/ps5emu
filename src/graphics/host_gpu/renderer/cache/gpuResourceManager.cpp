#include "graphics/host_gpu/renderer/cache/gpuResourceManager.h"

#include "common/assert.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
namespace Libs::Graphics {

GpuResourceManager::GpuResourceManager(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_buffer_cache(graphics, scheduler, m_page_manager, m_texture_cache, m_resource_mutex),
      m_texture_cache(graphics, scheduler, m_page_manager, m_buffer_cache, m_resource_mutex) {}

GpuResourceManager::~GpuResourceManager() = default;

bool GpuResourceManager::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	constexpr uint64_t fault_size = 8;
	// A muro-18 guard page is deliberately NOT a mapped GPU resource, so the IsMapped
	// gate alone would filter its fault out before the page manager could disarm it and
	// resolve the watched successor.
	const bool guarded = m_page_manager.IsGuarded(fault_vaddr);
	if (!guarded && !IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported guest-memory fault from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " access=%u\n",
		     fault_vaddr, static_cast<uint32_t>(access));
	}
	bool       handled = false;
	const auto resolve = [this, access, fault_vaddr, guarded, &handled](CommandProcessor& cp) {
		cp.BeginReadbackTransaction();
#if defined(__APPLE__)
		// Muro 18, second pattern (restored after the guest-memory-rework rebase dropped
		// it): a sequential unaligned wide-store guest memcpy into a multi-page watched
		// run. Per-page resolution frees only the faulting page, so the store crossing
		// into the NEXT still-watched page aborts inside Rosetta with no deliverable
		// fault (same Apple bug as the range-edge case). Free the whole contiguous
		// watched run above the faulting page before the copy retries — and free it
		// TOP-DOWN: releasing bottom-up opens a window where a CONCURRENT copier's store
		// crosses from an already-freed page into a still-watched one. Descending order
		// keeps the invariant "no writable page sits below a watched one" at every step,
		// so a racing thread always faults cleanly at instruction start and waits. The
		// walk stops at the first unwatched page, so an armed guard above the run is
		// never chained into.
		const auto forward_resolve = [this, access, fault_vaddr, guarded]() {
			if (access != PageFaultAccess::Write) {
				return;
			}
			const auto page_size = m_page_manager.GetPageSize();
			const auto run_low = (fault_vaddr & ~(page_size - 1)) + page_size * (guarded ? 2 : 1);
			auto       run_end = run_low;
			while (m_page_manager.IsWatched(run_end)) {
				run_end += page_size;
			}
			uint64_t resolved = 0;
			for (auto addr = run_end; addr > run_low;) {
				addr -= page_size;
				(void)m_buffer_cache.SynchronizeBacking(addr, 1);
				ResourceMutex::FaultScope forward_fault(m_resource_mutex);
				m_buffer_cache.InvalidateMemory(addr, 1);
				m_texture_cache.InvalidateMemory(addr, 1);
				if (m_page_manager.IsWatched(addr)) {
					// A page that stays watched after both invalidations would leave the
					// pages freed below it sitting writable under a watched one — the
					// exact geometry this walk exists to prevent. Stop here so the run
					// keeps its safe floor (mirrors the break-on-failure of the original
					// per-page resolution).
					std::fprintf(stderr,
					             "forward-resolve: page 0x%016llx stayed watched, "
					             "stopping run walk\n",
					             static_cast<unsigned long long>(addr));
					break;
				}
				resolved++;
			}
			if (resolved >= 16) {
				std::fprintf(stderr,
				             "forward-resolve: freed %llu watched pages above 0x%016llx\n",
				             static_cast<unsigned long long>(resolved),
				             static_cast<unsigned long long>(fault_vaddr));
			}
		};
		// The run above the faulting page is freed FIRST (top-down): the faulting page
		// itself stays protected until the main resolution below, so a concurrent copier
		// can never find a writable page under a watched one while the run unwinds.
		forward_resolve();
#endif
		{
			ResourceMutex::FaultScope fault(m_resource_mutex);
			if (guarded) {
				// Disarm the guard and resolve the watched successor together so a
				// retried split store crosses two writable pages (muro 18).
				m_page_manager.DisarmGuard(fault_vaddr);
				const auto successor = fault_vaddr + m_page_manager.GetPageSize();
				if (access == PageFaultAccess::Write) {
					m_buffer_cache.InvalidateMemory(successor, fault_size);
					m_texture_cache.InvalidateMemory(successor, fault_size);
				} else {
					m_buffer_cache.ReadMemory(successor, fault_size);
				}
			}
			if (access == PageFaultAccess::Write) {
				m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
				m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
			} else {
				m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
			}
			handled = true;
		}
		cp.EndReadbackTransaction();
	};
	if (auto* cp = Gpu::CurrentCommandProcessor(); cp != nullptr) {
		resolve(*cp);
		return handled;
	}
	if (m_resource_mutex.IsOwnedByCurrentThread()) {
		EXIT("unsupported page fault from a pre-owned resource transaction, addr=0x%016" PRIx64
		     " access=%u\n",
		     fault_vaddr, static_cast<uint32_t>(access));
	}
	EXIT_IF(m_gpu == nullptr);
	m_gpu->SendCommandSyncWithProcessor(resolve);
	return handled;
}

bool GpuResourceManager::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory invalidation from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto resolve = [this, vaddr, size](CommandProcessor& cp) {
		cp.BeginReadbackTransaction();
		{
			ResourceMutex::FaultScope fault(m_resource_mutex);
			m_buffer_cache.InvalidateMemory(vaddr, size);
			m_texture_cache.InvalidateMemory(vaddr, size);
		}
		cp.EndReadbackTransaction();
	};
	if (auto* cp = Gpu::CurrentCommandProcessor(); cp != nullptr) {
		resolve(*cp);
		return true;
	}
	if (m_resource_mutex.IsOwnedByCurrentThread()) {
		EXIT("unsupported memory invalidation from a pre-owned resource transaction, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	EXIT_IF(m_gpu == nullptr);
	m_gpu->SendCommandSyncWithProcessor(resolve);
	return true;
}

bool GpuResourceManager::SynchronizeImageToMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size) || CommandScheduler::InDeferredOperation()) {
		return false;
	}
	bool       synchronized = false;
	const auto resolve      = [this, vaddr, size, &synchronized](CommandProcessor& cp) {
        cp.BeginReadbackTransaction();
        {
            ResourceMutex::FaultScope fault(m_resource_mutex);
            synchronized = m_texture_cache.SynchronizeImageToBuffer(vaddr, size);
        }
        cp.EndReadbackTransaction();
	};
	if (auto* cp = Gpu::CurrentCommandProcessor(); cp != nullptr) {
		resolve(*cp);
		return synchronized;
	}
	if (m_resource_mutex.IsOwnedByCurrentThread() || m_gpu == nullptr) {
		return false;
	}
	m_gpu->SendCommandSyncWithProcessor(resolve);
	return synchronized;
}

bool GpuResourceManager::SynchronizeBufferToMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size) || CommandScheduler::InDeferredOperation()) {
		return false;
	}
	// Widen the pull to whole tracker pages so a single download (one GPU drain) settles
	// every later read of the same table instead of paying one drain per dword.
	auto page_begin = vaddr & ~(TRACKER_PAGE_SIZE - 1);
	auto page_end   = (vaddr + size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
	if (!IsMapped(page_begin, page_end - page_begin)) {
		// The widened window can cross out of the mapped range at an allocation edge;
		// fall back to the exact range instead of assuming the guest's allocation
		// granularity keeps whole tracker pages mapped around every readable address.
		page_begin = vaddr;
		page_end   = vaddr + size;
	}
	if (!m_buffer_cache.IsRegionGpuModified(page_begin, page_end - page_begin)) {
		return false;
	}
	bool       synchronized = false;
	const auto resolve      = [this, page_begin, page_end, &synchronized](CommandProcessor& cp) {
        cp.BeginReadbackTransaction();
        {
            ResourceMutex::FaultScope fault(m_resource_mutex);
            m_buffer_cache.ReadMemory(page_begin, page_end - page_begin);
            synchronized = true;
        }
        cp.EndReadbackTransaction();
	};
	if (auto* cp = Gpu::CurrentCommandProcessor(); cp != nullptr) {
		resolve(*cp);
		return synchronized;
	}
	if (m_resource_mutex.IsOwnedByCurrentThread() || m_gpu == nullptr) {
		return false;
	}
	m_gpu->SendCommandSyncWithProcessor(resolve);
	return synchronized;
}

bool GpuResourceManager::IsBufferRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return IsMapped(vaddr, size) && m_buffer_cache.IsRegionGpuModified(vaddr, size);
}

bool GpuResourceManager::IsBufferRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return IsMapped(vaddr, size) && m_buffer_cache.IsRegionCpuModified(vaddr, size);
}

bool GpuResourceManager::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (vaddr == 0 || size == 0 || vaddr >= TRACKER_ADDRESS_SIZE ||
	    size > TRACKER_ADDRESS_SIZE - vaddr) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void GpuResourceManager::MapMemory(uint64_t vaddr, uint64_t size) {
	{
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Add(vaddr, size);
	}
	m_page_manager.OnGpuMap(vaddr, size);
}

void GpuResourceManager::UnmapMemory(uint64_t vaddr, uint64_t size) {
	const auto unmap = [this, vaddr, size] {
		m_buffer_cache.UnmapMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_page_manager.OnGpuUnmap(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	if (m_gpu == nullptr) {
		if (m_resource_mutex.IsOwnedByCurrentThread()) {
			EXIT("cannot synchronously unmap from a resource transaction\n");
		}
		unmap();
		return;
	}
	const bool gpu_mapped = [&] {
		std::shared_lock lock(m_mapped_ranges_mutex);
		return m_mapped_ranges.Intersects(vaddr, size);
	}();
	// A range that never went through MapMemory has nothing cached the GPU could still
	// reference, so skip the submission pause and full GPU drain (muro 40). Callers
	// hold the global memory-operation mutex, and waiting for the GPU under it
	// deadlocks against guest label writers the GPU is itself waiting on (three-way
	// stall: memory op -> pause -> WaitRegMem -> guest writer blocked on the memory op).
	if (!gpu_mapped) {
		if (m_resource_mutex.IsOwnedByCurrentThread()) {
			EXIT("cannot synchronously unmap from a resource transaction\n");
		}
		// The short path rests on "never MapMemory'd => nothing cached". Verify it and
		// fail loud on a violation instead of silently re-creating the cross-thread
		// wait this branch exists to avoid.
		const bool cached = m_buffer_cache.HasPageOverlap(vaddr, size) ||
		                    m_texture_cache.QueryRegion(vaddr, size).image_pages;
		if (cached) {
			EXIT("gpu-unmapped range still holds cached resources: addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 "\n",
			     vaddr, size);
		}
		unmap();
		return;
	}
	Gpu::SubmissionLock submissions(*m_gpu);
	m_gpu->SendCommandSync(unmap);
}

void GpuResourceManager::RunGarbageCollector() {
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

} // namespace Libs::Graphics
