export interface OverlayConfig {
  WS_HOST: string;
  WS_PORT: number | null;
  WS_ENDPOINT: string;
  ALERT_BASE_DISPLAY_MS: number;
  TTS_VOLUME: number;
  TTS_RATE: number;
  SOUND_VOLUME: number;
  LOCK_TIMEOUT_MS: number;
  SOUND_RETRY_MAX: number;
  SOUND_RETRY_DELAY_MS: number;
  SOUND_CHECK_TIMEOUT_MS: number;
  SKIP_SILENCE_MS: number;
  SITE_ROOT: string;
  SOUND_FILES: {
    common: {
      error: string;
      unknown: string;
    };
  };
}

export const OVERLAY_CONFIG: OverlayConfig = {
  WS_HOST: "127.0.0.1",
  WS_PORT: null,
  WS_ENDPOINT: "/",
  ALERT_BASE_DISPLAY_MS: 7000,
  TTS_VOLUME: 0.9,
  TTS_RATE: 1.0,
  SOUND_VOLUME: 1.0,
  LOCK_TIMEOUT_MS: 15 * 1000,
  SOUND_RETRY_MAX: 5,
  SOUND_RETRY_DELAY_MS: 500,
  SOUND_CHECK_TIMEOUT_MS: 300,
  SKIP_SILENCE_MS: 5000,
  SITE_ROOT: "/",
  SOUND_FILES: {
    common: {
      error: "../../_shared/error_alert.mp3",
      unknown: "../../_shared/unknown_alert.mp3",
    },
  },
};
