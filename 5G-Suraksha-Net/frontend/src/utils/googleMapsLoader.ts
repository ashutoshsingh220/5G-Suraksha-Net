/**
 * Dynamic, asynchronous Google Maps JavaScript API loader.
 * Reuses existing script tags and caches the promise across component remounts.
 */

declare global {
  interface Window {
    google?: any;
  }
}

let googleMapsPromise: Promise<any> | null = null;

export function loadGoogleMaps(apiKey?: string): Promise<any> {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Window undefined in SSR'));
  }

  if (window.google?.maps) {
    return Promise.resolve(window.google);
  }

  const key =
    apiKey ||
    import.meta.env.VITE_GOOGLE_MAPS_API_KEY ||
    '';

  if (!key) {
    return Promise.reject(new Error('Google Maps API key is not configured'));
  }

  if (googleMapsPromise) {
    return googleMapsPromise;
  }

  googleMapsPromise = new Promise((resolve, reject) => {
    const existing = document.getElementById('google-maps-js-sdk');
    if (existing) {
      if (window.google?.maps) {
        resolve(window.google);
      } else {
        existing.addEventListener('load', () => resolve(window.google));
        existing.addEventListener('error', (err) => reject(err));
      }
      return;
    }

    const script = document.createElement('script');
    script.id = 'google-maps-js-sdk';
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(
      key
    )}&libraries=places,geometry`;
    script.async = true;
    script.defer = true;

    script.onload = () => {
      if (window.google?.maps) {
        resolve(window.google);
      } else {
        reject(new Error('google.maps not available after script load'));
      }
    };

    script.onerror = (err) => {
      googleMapsPromise = null;
      reject(err);
    };

    document.head.appendChild(script);
  });

  return googleMapsPromise;
}
