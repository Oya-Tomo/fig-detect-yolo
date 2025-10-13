import datetime
import hashlib
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Self

import fitz
import requests
from PIL import Image
from pydantic import BaseModel


def xml_to_json(xml: ET.Element):
    # Convert XML to JSON
    def expand_children(node: ET.Element):
        obj = {
            "tag": node.tag.split("}")[1],
            "tag_prefix": re.findall(r"{.*}", node.tag)[0][1:-1],
            "text": node.text,
            "attributes": node.attrib,
            "children": [],
        }

        for child in node:
            obj["children"] = obj.get("children", []) + [expand_children(child)]
        return obj

    return expand_children(xml)


def xml_to_arxiv_json(xml: ET.Element):
    data = xml_to_json(xml)

    if data["tag"] != "feed":
        raise ValueError("Expected root tag to be 'feed'")

    entries = []

    for child in data["children"]:
        if child["tag"] == "entry":
            entry = {
                "id": None,
                "title": None,
                "summary": None,
                "updated": None,
                "published": None,
                "authors": [],
                "pdf": None,
                "comment": None,
                "journal_ref": None,
                "doi": None,
                "primary_category": None,
                "categories": [],
            }
            for entry_child in child["children"]:
                if entry_child["tag"] == "id":
                    entry["id"] = entry_child["text"]
                elif entry_child["tag"] == "title":
                    entry["title"] = entry_child["text"]
                elif entry_child["tag"] == "summary":
                    entry["summary"] = entry_child["text"]
                elif entry_child["tag"] == "updated":
                    entry["updated"] = entry_child["text"]
                elif entry_child["tag"] == "published":
                    entry["published"] = entry_child["text"]
                elif entry_child["tag"] == "author":
                    author = {
                        "name": None,
                        "affiliation": None,
                    }
                    for author_child in entry_child["children"]:
                        if author_child["tag"] == "name":
                            author["name"] = author_child["text"]
                        elif author_child["tag"] == "affiliation":
                            author["affiliation"] = author_child["text"]
                    entry["authors"].append(author)
                elif entry_child["tag"] == "link":
                    if entry_child["attributes"].get("type", None) == "application/pdf":
                        entry["pdf"] = entry_child["attributes"]["href"]
                elif entry_child["tag"] == "comment":
                    entry["comment"] = entry_child["text"]
                elif entry_child["tag"] == "journal_ref":
                    entry["journal_ref"] = entry_child["text"]
                elif entry_child["tag"] == "doi":
                    entry["doi"] = entry_child["text"]
                elif entry_child["tag"] == "primary_category":
                    entry["primary_category"] = entry_child["attributes"]["term"]
                elif entry_child["tag"] == "category":
                    entry["categories"].append(entry_child["attributes"]["term"])
            entries.append(entry)
    return entries


def url_to_arxiv_id(url: str) -> str | None:
    id_part = re.match(r"^(http|https)://arxiv.org/abs/([a-zA-Z0-9.\-/]*)$", url)
    if id_part is None:
        return None
    version_part = re.search(r"v\d+$", id_part.group(2))
    if version_part is None:
        return id_part.group(2)
    else:
        return id_part.group(2)[: -len(version_part.group(0))]


class ArxivPaper(BaseModel):
    id: str
    src: str = "arxiv"

    title: str
    abstract: str

    authors: list[str]
    organizations: list[str]

    url: str
    pdf: str

    journal: str | None
    doi: str | None

    published_at: datetime.datetime
    updated_at: datetime.datetime

    @classmethod
    def from_dict(self, data: dict) -> Self:
        return ArxivPaper(
            id=url_to_arxiv_id(data["id"]),
            title=data["title"],
            abstract=data["summary"],
            authors=[author["name"] for author in data["authors"]],
            organizations=list(
                set(author["affiliation"] for author in data["authors"]) - {None}
            ),
            url=data["id"],
            pdf=data["pdf"],
            journal=data["journal_ref"],
            doi=data["doi"],
            published_at=datetime.datetime.fromisoformat(data["published"]),
            updated_at=datetime.datetime.fromisoformat(data["updated"]),
        )


def collect_arxiv_papers(
    search_query: str | None = None,
    id_list: str | None = None,
    start: int = 0,
    max_results: int = 10,
) -> list[ArxivPaper]:
    base_url = f"http://export.arxiv.org/api/query?"
    params = []
    if search_query is not None:
        params.append(f"search_query={search_query}")
    if id_list is not None:
        params.append(f"id_list={id_list}")
    params.append(f"start={start}")
    params.append(f"max_results={max_results}")
    url = base_url + "&".join(params)

    response = requests.get(url)
    xml = ET.fromstring(response.text)
    data = xml_to_arxiv_json(xml)

    return [ArxivPaper.from_dict(paper) for paper in data]


def download_pdf(url: str, path: str) -> bool:
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        return True
    else:
        print(f"Failed to download PDF: {response.status_code}")
        return False


def generate_short_hash(seed: str) -> str:
    hash = hashlib.md5(seed.encode()).hexdigest()
    return hash


def get_pdf_page_images(pdf_path: str, dpi: int = 300) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    images = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return images


def main(max_results: int = 200):
    os.makedirs("dataset/papers", exist_ok=True)

    categories = [
        "cs.AI",
        "cs.AR",
        "cs.CC",
        "cs.CE",
        "cs.CG",
        "cs.CL",
        "cs.CR",
        "cs.CV",
        "cs.CY",
        "cs.DB",
        "cs.DC",
        "cs.DL",
        "cs.DM",
        "cs.DS",
        "cs.ET",
        "cs.FL",
        "cs.GL",
        "cs.GR",
        "cs.GT",
        "cs.HC",
        "cs.IR",
        "cs.IT",
        "cs.LG",
        "cs.LO",
        "cs.MA",
        "cs.MM",
        "cs.MS",
        "cs.NA",
        "cs.NE",
        "cs.NI",
        "cs.OH",
        "cs.OS",
        "cs.PF",
        "cs.PL",
        "cs.RO",
        "cs.SC",
        "cs.SD",
        "cs.SE",
        "cs.SI",
        "cs.SY",
        "physics.acc-ph",
        "physics.ao-ph",
        "physics.app-ph",
        "physics.atm-clus",
        "physics.atom-ph",
        "physics.bio-ph",
        "physics.chem-ph",
        "physics.class-ph",
        "physics.comp-ph",
        "physics.data-an",
        "physics.ed-ph",
        "physics.flu-dyn",
        "physics.gen-ph",
        "physics.geo-ph",
        "physics.hist-ph",
        "physics.ins-det",
        "physics.med-ph",
        "physics.optics",
        "physics.plasm-ph",
        "physics.pop-ph",
        "physics.soc-ph",
        "physics.space-ph",
        "q-bio.BM",
        "q-bio.CB",
        "q-bio.GN",
        "q-bio.MN",
        "q-bio.NC",
        "q-bio.OT",
        "q-bio.PE",
        "q-bio.QM",
        "q-bio.SC",
        "q-bio.TO",
        "eess.AS",
        "eess.IV",
        "eess.SP",
        "eess.SY",
    ]

    queries = [f"cat:{cat}" for cat in categories]
    id_list = None
    start = 0

    print(f"Starting downloading papers...")
    for query in queries:
        print(f"Query: {query}")
        papers = collect_arxiv_papers(
            search_query=query,
            id_list=id_list,
            start=start,
            max_results=max_results,
        )
        print(f"    Found {len(papers)} papers")
        for paper in papers:
            print(f"    {paper.id} - {paper.title}")
            paper_hash = generate_short_hash(paper.id)
            pdf_path = f"dataset/papers/{paper_hash}.pdf"
            if os.path.exists(pdf_path):
                print(f"        PDF already exists, skipping download")
                continue
            if not download_pdf(paper.pdf, pdf_path):
                print(f"        Failed to download PDF")
                continue
            print(f"        Downloaded PDF to {pdf_path}")
            time.sleep(3)  # To respect arXiv's rate limits


if __name__ == "__main__":
    main(max_results=200)
