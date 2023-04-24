from bs4 import BeautifulSoup
import os
import glob

# List of URLs to remove
nsfw_links = [
]

folders = ['visualize/laion']
for folder in folders:
    html_files = glob.glob(os.path.join(folder, '*.html'))
    for html_file in html_files:
        print(f"Cleaning {html_file}...")
        # Read the HTML content from the input file
        with open(html_file, "r") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        # Find all img tags
        img_tags = soup.find_all("img")

        for img in img_tags:
            if img["src"] in nsfw_links:
                img.decompose()

        # Write the modified HTML content to the output file
        with open(html_file, "w") as f:
            f.write(soup.prettify())