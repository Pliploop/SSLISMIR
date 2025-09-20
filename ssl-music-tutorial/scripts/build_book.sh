
pip install pyppeteer
pip install jupyter-book
pip install sphinxcontrib-mermaid
pip install ghp-import
pip install pytest-playwright

playwright install

jupyter-book build ./book
# jupyter-book build book/ --builder pdf

cd book
ghp-import -n -p -f _build/html