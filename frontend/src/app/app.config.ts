import {
  APP_INITIALIZER,
  ApplicationConfig,
  isDevMode,
  provideBrowserGlobalErrorListeners,
  provideZonelessChangeDetection,
} from '@angular/core';
import { provideRouter, withViewTransitions } from '@angular/router';
import { provideClientHydration, withEventReplay } from '@angular/platform-browser';
import { HttpClient, provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';

import { routes } from './app.routes';
import { apiKeyInterceptor } from './core/interceptors/api-key.interceptor';
import { healthCheckInterceptor } from './core/interceptors/health-check.interceptor';
import { API_BASE_URL } from './core/tokens/api-base-url.token';

// Runtime config holder — populated by loadRuntimeConfig before bootstrap.
const runtimeConfig: { apiBaseUrl: string } = {
  apiBaseUrl: isDevMode() ? 'http://localhost:8000/api/v1' : 'https://baudo-spoon.onrender.com/api/v1',
};

function loadRuntimeConfig(http: HttpClient): () => Promise<void> {
  // In dev, skip fetching config.json so localhost works without a static server.
  if (isDevMode()) return () => Promise.resolve();
  return () =>
    http
      .get<{ apiBaseUrl: string }>('/config.json')
      .toPromise()
      .then(cfg => {
        if (cfg?.apiBaseUrl) runtimeConfig.apiBaseUrl = cfg.apiBaseUrl;
      })
      .catch(() => {
        // Fallback to compiled default if config.json is unreachable.
      });
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZonelessChangeDetection(),
    provideRouter(routes, withViewTransitions()),
    provideClientHydration(withEventReplay()),
    provideHttpClient(withFetch(), withInterceptors([apiKeyInterceptor, healthCheckInterceptor])),
    {
      provide: APP_INITIALIZER,
      useFactory: loadRuntimeConfig,
      deps: [HttpClient],
      multi: true,
    },
    { provide: API_BASE_URL, useFactory: () => runtimeConfig.apiBaseUrl },
  ],
};
