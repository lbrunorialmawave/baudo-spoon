import { Component, input } from '@angular/core';

@Component({
  selector: 'app-skeleton',
  standalone: true,
  template: `
    <div
      class="skeleton w-full rounded-lg"
      [style.height]="height()"
      role="status"
      aria-label="Loading..."
    ></div>
  `,
})
export class SkeletonComponent {
  readonly height = input<string>('1rem');
}
