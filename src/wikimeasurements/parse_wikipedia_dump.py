import os
import re
import json
import time
import logging
from datetime import timedelta
from bz2 import BZ2File
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
from tqdm import tqdm
import mwparserfromhell
from wikimeasurements.utils.general_utils import init_logger
from mwparserfromhell.nodes.template import Template
from wikimeasurements.utils.mediawiki_utils import (
    get_wikipedia_statistics,
    clean_wiki_page,
    parse_convert_template,
    parse_mediawiki_markup,
)


logger = init_logger("logs/parse_wikipedia_dump.log", logging.DEBUG)

class WikipediaDumpParser:
    """Parser for Wikipedia XML dumps. The parser extracts page content and optionally expands convert templates.
    The basic Wikipedia XML parse logic is based on https://github.com/TaherAmlaki/ParsingWikipediaXml/blob/main/etParser.py.
    It was modified to handle revisions, page titles, page IDs, and not loose elem.text from start event.
    """
    def __init__(self, xml_dump_path: str, output_file_path: str, output_mode: str="raw", limit: int=None, page_offset: int=0):
        logger.info(f"Using Wikipedia dump located at {xml_dump_path}")
        self.xml_dump = BZ2File(xml_dump_path)
        self.output_file = open(output_file_path, "w", encoding="utf-8")
        self.output_mode = output_mode
        self.debug_mode = False
        self.limit = limit
        self.page_offset = page_offset
        self.dump_filename = os.path.basename(xml_dump_path)
        self.wiki_lang = self.dump_filename.split("wiki")[0]
        self.approx_total_page_count = get_wikipedia_statistics(lang=self.wiki_lang, print_stats=True)["articles"]        
        convert_regex = r"({{(?:convert|cvt)\|.+?(?=}})}})"
        self.convert_pattern = re.compile(convert_regex, flags=re.IGNORECASE | re.DOTALL)


    def parse(self):
        """Parses Wikipedia XML dump (tested with XML dump from Q3 2022)."""

        pbar = tqdm(total=self.approx_total_page_count, desc="Processing Wikipedia pages", unit="pages")
        page_id = None
        page_title = None
        page_content = None
        page_namespace = None
        tags_stack = None
        inside_revision = False
        context = ET.iterparse(self.xml_dump, events=("start", "end"))
        start_time = time.time()

        for event, elem in context:

            tag_name = elem.tag.rsplit("}", 1)[-1].strip()

            if event == "start":
                if tag_name == "page":
                    # Initialize page.
                    page_id = None
                    page_title = ""
                    page_content = ""
                    page_namespace = 0
                    tags_stack = []                   
                    inside_revision = False
                elif tag_name == "revision":                    
                    inside_revision = True

                if page_content != None:
                    tags_stack.append(tag_name)

            else:
                if page_content != None:
                    # Inside page tags
                    if elem.text != None:
                        if tags_stack[-1] == "text":
                            # Append text to page content.
                            page_content += elem.text
                        elif tags_stack[-1] == "title":
                            # Set page title.
                            page_title = elem.text
                        elif tags_stack[-1] == "ns":
                            # Set page namespace.
                            try:
                                page_namespace = int(elem.text)
                            except:
                                page_namespace = None
                        elif tags_stack[-1] == "id" and not inside_revision:
                            # Set page ID.
                            try:
                                page_id = int(elem.text)
                            except:
                                page_id = None

                    if tags_stack[-1] == "page":
                        # End of page.
                        if (page_content != None and page_namespace not in [None, 0]):
                            
                            if pbar.n >= self.page_offset:
                                # Process complete page content.
                                new_line = self.process_page(page_id, page_title, page_content)
                                if new_line != None:
                                    self.output_file.write(new_line)

                            # Update progress bar
                            pbar.update(1)

                            # Print progress once in a while.
                            if pbar.n % 10000 == 0:
                                execution_time = time.time() - start_time
                                processing_speed = pbar.n / execution_time
                                remainig_time = (self.approx_total_page_count - pbar.n) / processing_speed
                                logger.info(
                                        f"⏱️ {pbar.n} pages have been processed in {str(timedelta(seconds=execution_time))} h:min:s.\n" \
                                        f"    The expected remaining time is {str(timedelta(seconds=remainig_time))} h:min:s."
                                    )

                                if self.limit is not None and (pbar.n - self.page_offset) >= self.limit:
                                    logger.info("Specified limit reached. Stopping parsing.")
                                    break
        
                        page_content = None
                        tags_stack = None
                    else:                                   
                        del tags_stack[-1]

                # Clear elem only after end event in order 
                # to not lose elem.text from start event.
                elem.clear()

        pbar.close()

    def process_page(self, id, title, page):
        """Processes a Wikipedia page."""
       
        redirect_pattern = re.compile("#REDIRECT", re.IGNORECASE)
        if bool(redirect_pattern.match(page)):
            row = None
        else:
            try:
                if self.output_mode == "raw":
                    # Create raw output.                
                    row = json.dumps({"id": id, "title": title, "text": page}, ensure_ascii=False)                  
                elif self.output_mode == "convert":
                    # Create output with convert template expanded.                    
                    quantities, other_text = self.parse_page_with_convert_template(page)
                    row = json.dumps(
                        {"id": id, "title": title, "text": other_text, "quantities": quantities},
                        ensure_ascii=False,
                    )
                    if len(quantities) > 0:
                        logger.info(f"Found {len(quantities)} {'quantities' if len(quantities) > 1 else 'quantity'}.")

                row += "\n"
                
            except Exception as e:
                print(f"Error processing page {title} ({id}): {e}")
                row = None
  
        return row


    def split_at_convert_template(self, page: str):
        """Split page into convert templates and other content. 
        Every second element is an expanded convert template and 
        every other element is the text between convert templates.
        """

        if self.convert_pattern.search(page) is None:
            # No convert template found in page
            page_split = [page]
        else:
            # Alternately, list other text and convert templates
            wikicode = mwparserfromhell.parse(page)
            page_split = [""]
            i = 0
            for node in wikicode.nodes:
                if type(node) == Template and node.name.lower() in ["convert", "cvt"]:
                    page_split.extend([str(node), ""])
                    i += 2
                else:
                    page_split[i] += str(node)

        return page_split


    def parse_page_with_convert_template(self, page):

        other_text = []
        quantities = []

        # Clean page, that is, remove infoboxes, comments, etc.
        clean_page = clean_wiki_page(page)

        # Split page into convert templates and other content
        splits = self.split_at_convert_template(clean_page)

        for i, text in enumerate(splits):

            if i % 2:
                # Process template
                quantity = parse_convert_template(text)
                quantities.append(quantity)

            else:
                # Process other text, that is, clean Mediawiki markup
                # TODO: Do not add "\n\n" anymore (e.g., in '"text": [["\n\n", "Analysis of...')
                clean_text = parse_mediawiki_markup(text, debug_mode=self.debug_mode)
                other_text.append(clean_text)

        return quantities, other_text


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--xml_dump_path",
        default="./workflow/input/enwiki-latest-pages-articles-multistream.xml.bz2",
        help="""Path to local Wikipedia XML dump (e.g., 
        'enwiki-latest-pages-articles-multistream.xml.bz2')""",
    )
    parser.add_argument(
        "--outfile",
        default="./workflow/intermediates/parsed_wikipedia_dump.json",
        help="""Path of output file (e.g., './parsed_wikipedia_dump.json')""",
    )
    parser.add_argument(
        "--output_mode",
        default="convert",
        help="""Choose between "convert" and "raw". 
        Choosing "raw" the page content is written to file as is (that is, as MediaWiki markup). 
        Choosing "convert" the MediaWiki markup is parsed and the convert template expanded.""",
    )
    parser.add_argument(
        "--page_offset",
        default=0,
        help="""Skip all pages until page offset.""",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="""Max. number of pages to parse.""",
    )

    # TODO: Parallelize    
    args = parser.parse_args()    
    wiki_dump_parser = WikipediaDumpParser(args.xml_dump_path, args.outfile, output_mode=args.output_mode, limit=args.limit, page_offset=args.page_offset)
    wiki_dump_parser.parse()

    logger.info("Finished all tasks! ✔️")
