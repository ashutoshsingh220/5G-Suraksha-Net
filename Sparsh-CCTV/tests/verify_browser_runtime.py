import os
import sys
import time
import subprocess
import requests
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ARTIFACT_DIR = r"C:\Users\vinay\.gemini\antigravity-cli\brain\2379da73-cb8b-4a18-aab7-386bc5596e10"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://127.0.0.1:8000"

def wait_for_server(url, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{url}/api/system/health", timeout=1.0)
            if r.status_code == 200:
                print(f"[Verifier] Server healthy at {url}")
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def run_browser_verification():
    print("[Verifier] Starting FastAPI server on port 8000...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    try:
        if not wait_for_server(BASE_URL):
            raise RuntimeError("FastAPI server failed to start within timeout.")

        console_errors = []
        console_logs = []

        with sync_playwright() as p:
            print(f"[Verifier] Launching Google Chrome from: {CHROME_PATH}")
            browser = p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True,
                args=["--no-sandbox", "--disable-gpu"]
            )
            context = browser.new_context(viewport={"width": 1400, "height": 900})
            page = context.new_page()

            def on_console(msg):
                console_logs.append(f"[{msg.type}] {msg.text}")
                if msg.type in ("error", "assert") and "favicon.ico" not in msg.text:
                    console_errors.append(msg.text)

            def on_page_error(err):
                console_errors.append(str(err))

            page.on("console", on_console)
            page.on("pageerror", on_page_error)
            page.on("requestfailed", lambda req: print(f"[Verifier Req Failed] {req.url} {req.failure}"))
            page.on("response", lambda res: print(f"[Verifier Res {res.status}] {res.url}") if res.status >= 400 else None)

            print("[Verifier] Navigating to http://127.0.0.1:8000/dashboard ...")
            page.goto(f"{BASE_URL}/dashboard", wait_until="domcontentloaded")

            # 1. Observe Initial RTSP State for 4 seconds
            print("[Verifier] Observing initial RTSP state for 4 seconds...")
            page.wait_for_timeout(4000)

            print(f"[Verifier] Console logs so far ({len(console_logs)}):")
            for cl in console_logs:
                print("   ", cl)

            # Check that liveStreamImg is visible
            img_el = page.locator("#liveStreamImg")
            assert img_el.is_visible(), "liveStreamImg should be visible!"
            box = img_el.bounding_box()
            print(f"[Verifier] Video image rendered dimensions: {box['width']}x{box['height']}")
            assert box["width"] > 200 and box["height"] > 150, "Stream image must have non-zero dimensions"

            # Check feedSourceSelect dropdown is present and enabled
            sel_el = page.locator("#feedSourceSelect")
            assert sel_el.is_visible(), "feedSourceSelect must be visible"
            assert sel_el.is_enabled(), "feedSourceSelect must be ENABLED and openable"

            state = page.evaluate("() => ({ badge: document.getElementById('activeSourceBadge')?.innerText, selectCount: document.getElementById('feedSourceSelect')?.options?.length, selectVal: document.getElementById('feedSourceSelect')?.value })")
            print("[Verifier] In-page DOM state:", state)

            badge_text = page.locator("#activeSourceBadge").inner_text()
            print(f"[Verifier] Initial badge text: '{badge_text}'")
            assert "RTSP" in badge_text, f"Expected RTSP in badge, got '{badge_text}'"

            # Capture Screenshot 1: RTSP Slate
            path1 = os.path.join(ARTIFACT_DIR, "verification_rtsp_slate.png")
            page.screenshot(path=path1)
            print(f"[Verifier] Saved Screenshot 1: {path1}")

            # 2. Select crash.mp4 and Switch Feed
            print("[Verifier] Switching feed source to crash.mp4 ...")
            page.select_option("#feedSourceSelect", "crash.mp4")
            page.click("button:has-text('Switch Feed')")

            # Observe test video playback for 4 seconds
            print("[Verifier] Observing crash.mp4 detection stream for 4 seconds...")
            page.wait_for_timeout(4000)

            badge_text2 = page.locator("#activeSourceBadge").inner_text()
            print(f"[Verifier] Updated badge text: '{badge_text2}'")
            assert "crash.mp4" in badge_text2, f"Expected crash.mp4 in badge, got '{badge_text2}'"

            fps_text = page.locator("#fpsVal").inner_text()
            print(f"[Verifier] Telemetry Stream Rate: {fps_text}")

            # Capture Screenshot 2: Crash test video playback
            path2 = os.path.join(ARTIFACT_DIR, "verification_test_video_crash.png")
            page.screenshot(path=path2)
            print(f"[Verifier] Saved Screenshot 2: {path2}")

            # 3. Switch to fire.mp4
            print("[Verifier] Switching feed source to fire.mp4 ...")
            page.select_option("#feedSourceSelect", "fire.mp4")
            page.click("button:has-text('Switch Feed')")
            page.wait_for_timeout(3500)

            badge_text3 = page.locator("#activeSourceBadge").inner_text()
            print(f"[Verifier] Fire badge text: '{badge_text3}'")
            assert "fire.mp4" in badge_text3, f"Expected fire.mp4 in badge, got '{badge_text3}'"

            # Capture Screenshot 3: Fire test video playback
            path3 = os.path.join(ARTIFACT_DIR, "verification_test_video_fire.png")
            page.screenshot(path=path3)
            print(f"[Verifier] Saved Screenshot 3: {path3}")

            # 4. Quick-Switch back to RTSP via "📡 Live RTSP" button
            print("[Verifier] Clicking '📡 Live RTSP' quick-toggle button...")
            page.click("#btnQuickRTSP")
            
            # Wait for RTSP switch probe to complete (up to 7 seconds)
            for _ in range(14):
                page.wait_for_timeout(500)
                badge_text4 = page.locator("#activeSourceBadge").inner_text()
                if "RTSP" in badge_text4:
                    break

            print(f"[Verifier] Returned badge text: '{badge_text4}'")
            assert "RTSP" in badge_text4, f"Expected RTSP in badge after quick return, got '{badge_text4}'"

            # Capture Screenshot 4: Return to RTSP
            path4 = os.path.join(ARTIFACT_DIR, "verification_return_rtsp.png")
            page.screenshot(path=path4)
            print(f"[Verifier] Saved Screenshot 4: {path4}")

            # 5. Switch to Dynamic Settings tab
            print("[Verifier] Clicking 'Dynamic Settings' tab...")
            page.click("button:has-text('Dynamic Settings')")
            page.wait_for_timeout(1000)

            settings_badge = page.locator("#settingsStatusBadge").inner_text()
            print(f"[Verifier] Settings badge text: '{settings_badge}'")
            assert "Synced" in settings_badge, f"Expected Synced badge, got '{settings_badge}'"

            cam_id_val = page.locator("#set_camera_id").input_value()
            cam_loc_val = page.locator("#set_camera_location").input_value()
            crash_conf_val = page.locator("#set_crash_conf").input_value()
            print(f"[Verifier] Settings inputs hydrated: camera_id='{cam_id_val}', location='{cam_loc_val}', crash_conf={crash_conf_val}")
            assert cam_id_val, "camera_id input must not be empty"
            assert cam_loc_val, "camera_location input must not be empty"

            # Capture Screenshot 5: Dynamic Settings hydrated
            path5 = os.path.join(ARTIFACT_DIR, "verification_settings_synced.png")
            page.screenshot(path=path5)
            print(f"[Verifier] Saved Screenshot 5: {path5}")

            # Verify console errors
            print(f"[Verifier] Total console logs recorded: {len(console_logs)}")
            print(f"[Verifier] Console errors recorded: {console_errors}")
            assert len(console_errors) == 0, f"Found JavaScript errors: {console_errors}"

            browser.close()
            print("\n🎉 ALL REAL BROWSER RUNTIME VERIFICATIONS PASSED SUCCESSFULLY!")

    finally:
        print("[Verifier] Terminating FastAPI server process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    run_browser_verification()
