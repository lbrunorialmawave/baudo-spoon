import { RenderMode, ServerRoute } from '@angular/ssr';

export const serverRoutes: ServerRoute[] = [
  { path: 'players',     renderMode: RenderMode.Client },
  { path: 'quotations',  renderMode: RenderMode.Client },
  { path: 'matches',     renderMode: RenderMode.Client },
  { path: 'predictions', renderMode: RenderMode.Client },
  { path: 'optimizer',  renderMode: RenderMode.Client },
  { path: 'auction',           renderMode: RenderMode.Client },
  { path: 'model-monitoring', renderMode: RenderMode.Client },
  { path: 'id-mapping',              renderMode: RenderMode.Client },
  { path: 'id-mapping/resolutions',  renderMode: RenderMode.Client },
  { path: 'admin',                   renderMode: RenderMode.Client },
  { path: '**',               renderMode: RenderMode.Client  },
];
