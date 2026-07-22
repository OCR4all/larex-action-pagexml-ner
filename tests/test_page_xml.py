from larex_action_pagexml_ner.main import (
    combine_lines,
    extract_page_text,
    safe_stem,
)


def test_extract_page_text_follows_nested_reading_order_and_text_equiv_index():
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">
      <Page imageWidth="100" imageHeight="100">
        <ReadingOrder>
          <OrderedGroup id="root">
            <OrderedGroupIndexed id="nested" index="0">
              <RegionRefIndexed regionRef="region-b" index="0"/>
            </OrderedGroupIndexed>
            <RegionRefIndexed regionRef="region-a" index="1"/>
          </OrderedGroup>
        </ReadingOrder>
        <TextRegion id="region-a">
          <TextLine id="line-a">
            <TextEquiv index="1"><Unicode>obsolete</Unicode></TextEquiv>
            <TextEquiv index="0"><Unicode>Second region</Unicode></TextEquiv>
          </TextLine>
        </TextRegion>
        <TextRegion id="region-b">
          <TextLine id="line-b"><TextEquiv><Unicode>First region</Unicode></TextEquiv></TextLine>
        </TextRegion>
      </Page>
    </PcGts>
    """

    extracted = extract_page_text(xml, dehyphenate=False)

    assert extracted.text == "First region\nSecond region"
    assert extracted.region_count == 2
    assert extracted.line_count == 2


def test_extract_page_text_appends_unreferenced_regions_and_uses_region_fallback():
    xml = b"""<PcGts xmlns="urn:page">
      <Page>
        <ReadingOrder>
          <OrderedGroup><RegionRefIndexed regionRef="region-b" index="0"/></OrderedGroup>
        </ReadingOrder>
        <TextRegion id="region-a">
          <TextEquiv><Unicode>Fallback text</Unicode></TextEquiv>
        </TextRegion>
        <TextRegion id="region-b">
          <TextLine><TextEquiv><Unicode>Referenced first</Unicode></TextEquiv></TextLine>
        </TextRegion>
      </Page>
    </PcGts>"""

    extracted = extract_page_text(xml, preserve_line_breaks=False)

    assert extracted.text == "Referenced first Fallback text"
    assert extracted.line_count == 2


def test_extract_page_text_normalizes_unicode_and_dehyphenates():
    xml = """<PcGts><Page><TextRegion id="r">
      <TextLine><TextEquiv><Unicode>Cafe\u0301 hy-</Unicode></TextEquiv></TextLine>
      <TextLine><TextEquiv><Unicode>phenation</Unicode></TextEquiv></TextLine>
      <TextLine><TextEquiv><Unicode>ISBN-</Unicode></TextEquiv></TextLine>
      <TextLine><TextEquiv><Unicode>123</Unicode></TextEquiv></TextLine>
    </TextRegion></Page></PcGts>""".encode()

    extracted = extract_page_text(xml, unicode_normalization="NFC")

    assert extracted.text == "Caf\u00e9 hyphenation\nISBN-\n123"


def test_combine_lines_can_flatten_without_dehyphenating():
    assert (
        combine_lines(["one-", "two"], preserve_line_breaks=False, dehyphenate=False) == "one- two"
    )


def test_safe_stem_removes_paths_and_unsafe_characters():
    assert safe_stem("../../Page 1?!?.xml") == "Page-1"
    assert safe_stem("...") == "page"
