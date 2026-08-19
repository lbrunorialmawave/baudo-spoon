import { Routes } from '@angular/router';
import { apiKeyGuard } from './core/guards/api-key.guard';
import { adminGuard } from './core/guards/role.guard';
import { ShellComponent } from './shared/components/shell/shell.component';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'players',
    pathMatch: 'full',
  },
  {
    path: '',
    component: ShellComponent,
    canActivate: [apiKeyGuard],
    children: [
      {
        path: 'players',
        loadComponent: () =>
          import('./features/players/players.component').then((m) => m.PlayersComponent),
      },
      {
        path: 'quotations',
        loadComponent: () =>
          import('./features/quotations/quotations.component').then((m) => m.QuotationsComponent),
      },
      {
        path: 'predictions',
        loadComponent: () =>
          import('./features/predictions/predictions.component').then(
            (m) => m.PredictionsComponent,
          ),
      },
      {
        path: 'overview',
        loadComponent: () =>
          import('./features/overview/overview.component').then((m) => m.OverviewComponent),
      },
      {
        path: 'optimizer',
        loadComponent: () =>
          import('./features/optimizer/optimizer.component').then((m) => m.OptimizerComponent),
      },
      {
        path: 'auction',
        loadComponent: () =>
          import('./features/auction/auction.component').then((m) => m.AuctionComponent),
      },
      {
        path: 'my-team',
        loadComponent: () =>
          import('./features/my-team/my-team.component').then((m) => m.MyTeamComponent),
      },
      {
        path: 'id-mapping',
        loadComponent: () =>
          import('./features/id-mapping/id-mapping.component').then((m) => m.IdMappingComponent),
      },
      {
        path: 'id-mapping/resolutions',
        loadComponent: () =>
          import('./features/id-mapping/resolution-history.component').then(
            (m) => m.ResolutionHistoryComponent
          ),
      },
      {
        path: 'admin',
        canActivate: [adminGuard],
        loadComponent: () =>
          import('./features/admin/admin.component').then((m) => m.AdminComponent),
      },
      {
        path: 'model-monitoring',
        canActivate: [adminGuard],
        loadComponent: () =>
          import('./features/model-monitoring/model-monitoring.component').then(
            (m) => m.ModelMonitoringComponent,
          ),
      },
    ],
  },
  {
    path: 'setup',
    loadComponent: () => import('./features/setup/setup.component').then((m) => m.SetupComponent),
  },
  {
    path: '**',
    loadComponent: () =>
      import('./features/not-found/not-found.component').then((m) => m.NotFoundComponent),
  },
];
