import {
  ApplicationConfig,
  isDevMode,
  provideBrowserGlobalErrorListeners,
  provideZonelessChangeDetection,
} from '@angular/core';
import { provideRouter, withViewTransitions } from '@angular/router';
import { provideClientHydration, withEventReplay } from '@angular/platform-browser';
import { provideHttpClient, withFetch, withInterceptors } from '@angular/common/http';

import { routes } from './app.routes';
import { apiKeyInterceptor } from './core/interceptors/api-key.interceptor';
import { API_BASE_URL } from './core/tokens/api-base-url.token';

function apiBaseUrlFactory(): string {
  // Use the Render API in production mode; keep localhost for dev.
  if (isDevMode()) {
    return 'http://localhost:8000/api/v1';
  }
  return 'https://baudo-spoon.onrender.com/api/v1';
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZonelessChangeDetection(),
    provideRouter(routes, withViewTransitions()),
    provideClientHydration(withEventReplay()),
    provideHttpClient(withFetch(), withInterceptors([apiKeyInterceptor])),
    { provide: API_BASE_URL, useFactory: apiBaseUrlFactory },
  ],
};
