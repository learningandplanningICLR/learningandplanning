from time import sleep

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from ourlib.logger import logger


class AwarelibVideoRecorder(VideoRecorder):
    # INFO: we added this because we want to modify the frame
    def capture_frame(self, transform_frame_fn=None):
        """Render the given `env` and add the resulting frame to the video."""
        if not self.functional: return
        logger.debug('Capturing video frame: path=%s', self.path)

        render_mode = 'ansi' if self.ansi_mode else 'rgb_array'
        # print('ja jebie mode = {}'.format(render_mode))
        # print(self.env)
        frame = self.env.render(mode=render_mode)

        # print(10 * '\n---------')
        # print(transform_frame_fn)
        # print(frame.shape)
        # print(render_mode)
        # sleep(10)

        if transform_frame_fn is not None:
            frame = transform_frame_fn(frame)

        if frame is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn('Env returned None on render(). Disabling further rendering for video recorder by marking as disabled: path=%s metadata_path=%s', self.path, self.metadata_path)
                self.broken = True
        else:
            self.last_frame = frame
            if self.ansi_mode:
                self._encode_ansi_frame(frame)
            else:
                self._encode_image_frame(frame)
