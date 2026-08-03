#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_

#include "common/common.h"

#include <memory>
#include <span>

namespace Libs::Graphics {

class CommandBuffer;
class RenderContext;
struct ImageInfo;
struct WindowContext;

class Presenter final {
public:
	struct Frame;

	explicit Presenter(WindowContext& window);
	~Presenter();
	KYTY_CLASS_NO_COPY(Presenter);

	[[nodiscard]] Frame&         PrepareFrame(CommandBuffer& command, const ImageInfo& info);
	[[nodiscard]] Frame&         PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
	                                               CommandBuffer* producer = nullptr);
	// m43g-intro probe (TEMPORARY, remove before commit): present host-side RGBA pixels
	// directly (the guest's own video-frame draws never happen at emulator speed).
	[[nodiscard]] Frame& PrepareHostFrame(const void* rgba, uint32_t width, uint32_t height);
	// m42-present probe (TEMPORARY, remove before commit): present the cache image at a
	// guest base address instead of the flip surface (G-buffer content is invisible to the
	// CPU readback, GPU->GPU copy is the only honest viewer). Falls back to the normal
	// surface when the address does not resolve or its format cannot feed the frame.
	// Takes a candidate list and acquires exactly ONE frame: acquiring per candidate would
	// drain the frame pool and deadlock the presenter.
	[[nodiscard]] Frame& PrepareCacheFrame(CommandBuffer& command, const ImageInfo& info,
	                                       std::span<const uint64_t> addresses,
	                                       uint64_t* hit_address, bool* substituted);
	[[nodiscard]] Frame*         PrepareLastFrame();
	[[nodiscard]] bool           IsGuestPaused() const noexcept;
	[[nodiscard]] RenderContext& Renderer() const noexcept;
	void                         Present(Frame& frame, bool reuse = false);
	void                         Discard(Frame& frame);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_PRESENTER_H_
