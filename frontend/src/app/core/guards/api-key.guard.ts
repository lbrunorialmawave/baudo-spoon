import { inject, PLATFORM_ID } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';
import { CanActivateFn, Router } from '@angular/router';

const API_KEY_STORAGE_KEY = 'fanta_api_key';

export const apiKeyGuard: CanActivateFn = () => {
  const router = inject(Router);
  const platformId = inject(PLATFORM_ID);

  // SSR: always allow through; guard is enforced client-side
  if (!isPlatformBrowser(platformId)) {
    return true;
  }

  const apiKey = localStorage.getItem(API_KEY_STORAGE_KEY);
  return apiKey ? true : router.createUrlTree(['/setup']);
};
