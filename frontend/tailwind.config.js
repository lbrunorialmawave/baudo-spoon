/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{html,ts,scss}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // FAANG-Spec Dynamic HSL Palette
        brand: {
          50:  'hsl(220, 100%, 97%)',
          100: 'hsl(220, 100%, 93%)',
          200: 'hsl(220, 100%, 85%)',
          300: 'hsl(220, 100%, 75%)',
          400: 'hsl(220, 100%, 65%)',
          500: 'hsl(220, 100%, 55%)',
          600: 'hsl(220, 100%, 45%)',
          700: 'hsl(220, 100%, 35%)',
          800: 'hsl(220, 100%, 25%)',
          900: 'hsl(220, 100%, 15%)',
        },
        surface: {
          DEFAULT: 'hsl(220, 15%, 10%)',
          raised: 'hsl(220, 15%, 13%)',
          overlay: 'hsl(220, 15%, 16%)',
          border: 'hsl(220, 15%, 22%)',
        },
        cluster: {
          0: '#6366f1', // indigo
          1: '#22d3ee', // cyan
          2: '#f59e0b', // amber
          3: '#10b981', // emerald
          4: '#f43f5e', // rose
          5: '#a855f7', // purple
          6: '#fb923c', // orange
          7: '#34d399', // teal
        },
      },
      fontFamily: {
        sans: ['Inter var', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'skeleton-pulse': 'skeleton-pulse 1.5s ease-in-out infinite',
        'fade-in': 'fade-in 200ms ease-out',
        'slide-up': 'slide-up 200ms ease-out',
      },
      keyframes: {
        'skeleton-pulse': {
          '0%, 100%': { opacity: '0.4' },
          '50%': { opacity: '0.8' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
    require('@tailwindcss/forms'),
  ],
};
