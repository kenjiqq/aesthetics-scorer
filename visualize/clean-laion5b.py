from bs4 import BeautifulSoup

# List of URLs to remove
nsfw_links = [
]

# Read the HTML content from the input file
with open("visualize/laion5b-visualize.html", "r") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

# Find all img tags
img_tags = soup.find_all("img")

for img in img_tags:
    if img["src"] in nsfw_links:
        img.decompose()

# Write the modified HTML content to the output file
with open("visualize/laion5b-visualize.html", "w") as f:
    f.write(soup.prettify())
