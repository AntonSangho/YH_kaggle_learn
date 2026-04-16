#!/usr/bin/env python3
"""
Fetch Kaggle Computer Vision course lesson and convert to markdown.
Usage: python fetch_cv_lesson.py 1   # fetch lesson 1
"""

import sys
import time
import base64
import re
import json
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from markdownify import markdownify as md
from bs4 import BeautifulSoup

# Lesson mapping
LESSONS = {
    1: {
        "title": "The Convolutional Classifier",
        "url": "https://www.kaggle.com/code/ryanholbrook/the-convolutional-classifier"
    },
    2: {
        "title": "Convolution and ReLU",
        "url": "https://www.kaggle.com/code/ryanholbrook/convolution-and-relu"
    },
    3: {
        "title": "Maximum Pooling",
        "url": "https://www.kaggle.com/code/ryanholbrook/maximum-pooling"
    },
    4: {
        "title": "The Sliding Window",
        "url": "https://www.kaggle.com/code/ryanholbrook/the-sliding-window"
    },
    5: {
        "title": "Custom Convnets",
        "url": "https://www.kaggle.com/code/ryanholbrook/custom-convnets"
    },
    6: {
        "title": "Data Augmentation",
        "url": "https://www.kaggle.com/code/ryanholbrook/data-augmentation"
    },
}

PROJECT_ROOT = Path(__file__).parent.parent
CV_DIR = PROJECT_ROOT / "ComputerVision"
IMG_DIR = CV_DIR / "img"

def extract_cells(page):
    """Extract notebook cells from the rendered notebook HTML."""
    cells = []

    print("Waiting for notebook to fully render...")
    time.sleep(2)
    page.wait_for_load_state("networkidle")
    time.sleep(1)

    try:
        # Extract content from the rendered notebook
        extract_script = """
        () => {
            const cells = [];

            // Find the main content container (usually with class 'border-box-sizing' or similar)
            let contentContainer = document.querySelector('.border-box-sizing') ||
                                   document.querySelector('[class*="notebook"]') ||
                                   document.body;

            // Get all relevant elements within the content container
            // We need to walk through all descendants while preserving order
            // Note: 'img' alone might not work in querySelectorAll order, so we get them separately
            const allElements = Array.from(contentContainer.querySelectorAll('h1, h2, h3, h4, h5, h6, p, pre, code, ul, ol, img'));

            let idx = 0;
            for (const elem of allElements) {
                // Handle IMG elements separately (they have no innerText)
                if (elem.tagName === 'IMG') {
                    cells.push({
                        index: idx++,
                        type: 'image',
                        src: elem.src,
                        alt: elem.alt || 'image'
                    });
                    continue;
                }

                // For other elements, skip if no text content
                if (!elem.innerText || elem.innerText.trim().length === 0) continue;

                const cellData = { index: idx++ };

                // Heading
                if (/^H[1-6]$/.test(elem.tagName)) {
                    const level = parseInt(elem.tagName[1]);
                    cellData.type = 'heading';
                    cellData.level = level;
                    cellData.content = elem.innerText.trim();
                }
                // Paragraph
                else if (elem.tagName === 'P') {
                    cellData.type = 'paragraph';
                    cellData.content = elem.innerText.trim();
                }
                // Code block
                else if (['PRE', 'CODE'].includes(elem.tagName)) {
                    cellData.type = 'code';
                    cellData.content = elem.innerText.trim();
                }
                // List
                else if (['UL', 'OL'].includes(elem.tagName)) {
                    cellData.type = 'list';
                    cellData.content = elem.innerText.trim();
                }

                if (cellData.content) {
                    cells.push(cellData);
                }
            }

            return cells;
        }
        """

        cells_data = page.evaluate(extract_script)
        print(f"Extracted {len(cells_data)} cells from notebook")

        for cell_data in cells_data:
            cell_type = cell_data.get('type', 'unknown')

            if cell_type == 'heading':
                # Create markdown heading
                level = cell_data.get('level', 1)
                content = cell_data.get('content', '')
                if content:
                    cells.append({
                        "type": "heading",
                        "content": f"{'#' * level} {content}",
                        "level": level
                    })
            elif cell_type in ('paragraph', 'list'):
                content = cell_data.get('content', '')
                if content and len(content) > 3:
                    cells.append({
                        "type": "text",
                        "content": content
                    })
            elif cell_type == 'code':
                content = cell_data.get('content', '')
                if content and len(content) > 3:
                    cells.append({
                        "type": "code",
                        "content": content
                    })
            elif cell_type == 'image':
                src = cell_data.get('src', '')
                if src:
                    cells.append({
                        "type": "image",
                        "content": src,
                        "alt": cell_data.get('alt', 'image')
                    })

    except Exception as e:
        print(f"Error in cell extraction: {e}")
        import traceback
        traceback.print_exc()

    print(f"Total cells extracted: {len(cells)}")
    return cells

def extract_images(html_content, output_dir, base_idx):
    """Extract images from HTML and save locally."""
    img_dir = Path(output_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    img_tags = re.findall(r'<img[^>]+>', html_content)

    for idx, img_tag in enumerate(img_tags):
        # Extract src
        src_match = re.search(r'src="([^"]+)"', img_tag)
        if not src_match:
            continue

        src = src_match.group(1)
        img_num = base_idx + idx + 1

        # Handle base64 data URIs
        if src.startswith('data:'):
            # Extract base64 content
            header_end = src.find(',')
            if header_end == -1:
                continue

            mime_info = src[5:header_end]
            base64_data = src[header_end+1:]

            # Decode and save
            try:
                img_bytes = base64.b64decode(base64_data)
                img_path = img_dir / f"{img_num}.png"
                with open(img_path, 'wb') as f:
                    f.write(img_bytes)
                image_paths.append((img_num, f"![](img/{img_num}.png)"))
                print(f"  Saved image {img_num}.png")
            except Exception as e:
                print(f"  Error decoding base64 image: {e}")
        else:
            # It's a URL - we could download it but for now just reference it
            # Or we might skip Kaggle-hosted images
            print(f"  Skipping URL image: {src[:50]}...")

    return image_paths

def process_cells(cells):
    """Process cells and convert to markdown."""
    markdown_parts = []
    img_counter = 0

    for cell in cells:
        cell_type = cell.get("type", "unknown")

        try:
            if cell_type == "markdown":
                # Convert HTML to markdown
                converted = md(cell["content"])
                if converted.strip():
                    markdown_parts.append(converted.strip())
                    print(f"  Added markdown cell")

            elif cell_type == "heading":
                # Already formatted as markdown heading
                content = cell["content"].strip()
                if content:
                    markdown_parts.append(content)
                    print(f"  Added heading")

            elif cell_type == "code":
                # Wrap code in fenced blocks
                code = cell["content"].strip()
                if code and len(code) > 3:
                    markdown_parts.append(f"```python\n{code}\n```")
                    print(f"  Added code block ({len(code)} chars)")

            elif cell_type == "output":
                # Extract images from output
                images = extract_images(cell["content"], IMG_DIR, img_counter)
                for img_num, img_ref in images:
                    img_counter = img_num
                    markdown_parts.append(img_ref)

            elif cell_type == "text":
                # Plain text extracted from HTML
                content = cell["content"].strip()
                if content and len(content) > 3:
                    markdown_parts.append(content)
                    print(f"  Added text ({len(content)} chars)")

            elif cell_type == "image":
                # Download and save image locally
                src = cell["content"]
                if src:
                    img_counter += 1
                    alt_text = cell.get('alt', 'image')

                    if src.startswith('data:'):
                        # Base64 image
                        try:
                            img_bytes = base64.b64decode(src.split(',')[1])
                            img_path = IMG_DIR / f"{img_counter}.png"
                            IMG_DIR.mkdir(parents=True, exist_ok=True)
                            with open(img_path, 'wb') as f:
                                f.write(img_bytes)
                            markdown_parts.append(f"![{alt_text}](img/{img_counter}.png)")
                            print(f"  Saved image {img_counter} (base64)")
                        except Exception as e:
                            print(f"  Error saving base64 image: {e}")
                    else:
                        # URL image - download it
                        try:
                            import urllib.request
                            img_path = IMG_DIR / f"{img_counter}.png"
                            IMG_DIR.mkdir(parents=True, exist_ok=True)

                            # Download the image
                            urllib.request.urlretrieve(src, img_path)
                            markdown_parts.append(f"![{alt_text}](img/{img_counter}.png)")
                            print(f"  Downloaded image {img_counter}")
                        except Exception as e:
                            print(f"  Error downloading image from {src[:50]}: {e}")
                            # Fallback to URL reference if download fails
                            markdown_parts.append(f"![{alt_text}]({src})")
                            print(f"  Fallback: referenced URL image")

            elif cell_type in ("raw", "fallback", "visible_text"):
                # Just use the raw text content
                content = cell["content"].strip()
                if content and len(content) > 10:
                    markdown_parts.append(content)
                    print(f"  Added {cell_type} ({len(content)} chars)")

        except Exception as e:
            print(f"  Error processing {cell_type} cell: {e}")

    return "\n\n".join(markdown_parts)

def fetch_lesson(lesson_num):
    """Fetch a single lesson and save to markdown."""
    if lesson_num not in LESSONS:
        print(f"Invalid lesson number. Available: {list(LESSONS.keys())}")
        return False

    lesson = LESSONS[lesson_num]
    print(f"\nFetching lesson {lesson_num}: {lesson['title']}")
    print(f"URL: {lesson['url']}")

    # Create output directory
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Launch browser and fetch
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            print("Loading Kaggle page...")
            page.goto(lesson['url'], wait_until="networkidle", timeout=30000)
            print("Page loaded, extracting notebook iframe...")

            # Get iframe URL
            iframe_url = page.evaluate("""
            () => {
                const iframe = document.querySelector('iframe');
                return iframe ? iframe.src : null;
            }
            """)

            if not iframe_url:
                print("Error: Could not find notebook iframe!")
                return False

            print(f"Found iframe, navigating to notebook content...")
            # Navigate to the iframe content directly
            page.goto(iframe_url, wait_until="networkidle", timeout=30000)
            print("Notebook content loaded")

            # Extract cells from the iframe
            cells = extract_cells(page)
            print(f"Extracted {len(cells)} cells")

            # Process cells to markdown
            content = process_cells(cells)

            # Add title
            title = lesson['title']
            final_markdown = f"# {title}\n\n{content}"

            # Save to file
            filename = f"{lesson_num}_{title.replace(' ', '')}.md"
            output_path = CV_DIR / filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_markdown)

            print(f"\n✓ Saved to {output_path}")
            return True

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            browser.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python fetch_cv_lesson.py <lesson_number>")
        print(f"Available lessons: {list(LESSONS.keys())}")
        sys.exit(1)

    try:
        lesson_num = int(sys.argv[1])
        success = fetch_lesson(lesson_num)
        sys.exit(0 if success else 1)
    except ValueError:
        print(f"Invalid lesson number: {sys.argv[1]}")
        sys.exit(1)

if __name__ == "__main__":
    main()
