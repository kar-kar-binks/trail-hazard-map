import asyncio
import shutil
import subprocess
import sys
from pathlib import Path

from playwright.async_api import async_playwright

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


async def wait_loaded(page):
    await page.wait_for_function("window.__app && window.__app.ready", timeout=120000)
    await page.evaluate("window.__app.settle()")
    await page.wait_for_timeout(2500)


async def shoot(page, frames, n=1):
    for _ in range(n):
        await page.screenshot(path=str(frames / f"f{len(list(frames.glob('f*.png'))):04d}.png"))


async def record(index_html, out_dir, area1, trail1, search_name, area2, trail2, walk_steps=36):
    out_dir = Path(out_dir)
    frames = out_dir / "_frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True, args=ARGS)
        page = await browser.new_page(viewport={"width": 1280, "height": 760})
        await page.goto("file://" + str(Path(index_html).resolve()))
        await wait_loaded(page)
        await page.select_option("#area", label=area1)
        await page.select_option("#trail", label=trail1)
        await wait_loaded(page)
        await page.evaluate("window.__app.setPos(window.__app.n*0.3)")
        await page.evaluate("window.__app.settle()")
        await shoot(page, frames, 6)
        await page.fill("#q", search_name)
        await shoot(page, frames, 3)
        await page.click("#go")
        await wait_loaded(page)
        await page.evaluate("window.__app.setPos(window.__app.n*0.3)")
        await page.evaluate("window.__app.settle()")
        await shoot(page, frames, 5)
        await page.select_option("#area", label=area2)
        await page.select_option("#trail", label=trail2)
        await wait_loaded(page)
        await page.evaluate("window.__app.setPos(0)")
        await page.evaluate("window.__app.settle()")
        await shoot(page, frames, 4)
        for k in range(walk_steps):
            f = k / (walk_steps - 1)
            await page.evaluate(f"window.__app.setPos(window.__app.n*{f:.4f}*0.985)")
            await page.evaluate("window.__app.settle()")
            await shoot(page, frames)
        await shoot(page, frames, 4)
        await browser.close()
    mp4 = out_dir / "demo.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "6", "-i", str(frames / "f%04d.png"), "-vf", "scale=1280:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(mp4)], check=True)
    gif = out_dir / "demo.gif"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "6", "-i", str(frames / "f%04d.png"), "-vf", "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3", str(gif)], check=True)
    shutil.rmtree(frames)
    return mp4, gif


if __name__ == "__main__":
    a = sys.argv
    asyncio.run(record(a[1], a[2], a[3], a[4], a[5], a[6], a[7]))
