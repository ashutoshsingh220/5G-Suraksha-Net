/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        suraksha: {
          bg: '#0B0F14',
          panel: '#111821',
          panelElevated: '#151E28',
          border: '#263341',
          blue: '#3B9EFF',
          cyan: '#49C6D9',
          green: '#2BC48A',
          amber: '#E7A83B',
          red: '#E05252',
          textPrimary: '#E8EDF3',
          textSecondary: '#98A6B5',
          textMuted: '#687585',
        }
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      boxShadow: {
        'panel': '0 4px 16px rgba(0, 0, 0, 0.25)',
        'subtle': '0 2px 8px rgba(0, 0, 0, 0.2)',
      },
      borderRadius: {
        card: '8px',
        btn: '6px',
        input: '6px',
        pill: '999px',
      },
      animation: {
        'pulse-subtle': 'pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
