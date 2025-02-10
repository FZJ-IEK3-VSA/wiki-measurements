import pytest
from wikimeasurements.utils.mediawiki_utils import parse_convert_template


def test_parse_convert_template_no_errors():
    """Test if processing an input convert template raises an error."""

    test_strings = [
        "{{convert|1234|ft|m|2}}",
        "{{convert|123.4|ft|m|abbr=off|sp=us}}",
        "{{convert|123.4|ft|m|abbr=off}}",
        "{{convert|4|ft||adj=mid|-long}}",
        "{{convert|40|acre||adj=pre|planted}}",
        "{{convert|4|m||disp=preunit|+ }}",
        "{{convert|4|m||disp=preunit|+ |or more }}",
        "{{convert|20|impfloz||disp=x|, approximately }}",
        "{{cvt|12x14x1|ft|cm|order=flip}}",
        "{{convert|100|m||disp=x|/day (|/day)}}",
        "{{convert|62.5|m|ft+royal cubit|0|abbr=on}}",
        "{{convert|100|km|0|adj=mid|in diameter|abbr=on}}",
        "{{convert|32.5|mpgus|L/100 km+mpgus+mpgimp|abbr=on|order=out}}",
        "{{convert|127000<ref>Bledsoe Architects</ref>|sqft|m2|adj=on}}",
        "{{Convert|20|fathom|4 = -deep|adj = mid}}",
        "{{Convert|120-150|C|4=-1|abbr=on|disp=comma}}",
        "{{convert|14|m|ft|adj=mid|[[Draft (hull)|draught]]}}",
        "{{convert|1|e12solar mass|kg|abbr=off|lk=on}}",
        "{{convert|0.8|+/-|0.1|e12solar mass|kg|abbr=on|lk=on}}",
        "{{Convert|30|km/s|4 = 1|abbr = on}}",
        "{{convert|305|acres|km2}}",
        "{{convert|100|km|0|adj=mid|in diameter|abbr=on}}",        
        "{{Convert|30|km/s|4 = 1|abbr = on}}",
        "{{convert|35|mi|4=}}",  # ignore last arg
        "{{convert|1⁄4|mi|m|-1|abbr=on}}",
        "{{convert|¼|mi|m|-1|abbr=on}}",
        "{{convert|²¹/₉₈₇|mi|m|-1|abbr=on}}",
        "{{cvt|1+ 3/8|in}}",
        "{{convert|3//4|in|mm}}",
        "{{convert|1.6|e6sqft|acre_m2}}",  # interprete underscore as whitespace
        "{{cvt|61.5|F|C|abbr=values|disp=x|/}}",
    ]
    for test_string in test_strings:
        parse_convert_template(test_string)


def test_parse_convert_template_output_correct():
    """Test if an input convert template is parsed into the correct text."""

    test_strings = [
        # ("{{convert|1|abbr=off|lk=on}}", None, ""),
        (
            "{{convert|127000<ref>Bledsoe Architects</ref>|sqft|m2|adj=on}}",
            "127,000-square-foot (11,800 m2)",
            "Louisiana_Tech_University",
        ),  # reference in Mediawiki output "127,000[33]-square-foot (11,800 m2)" is ignored here
        ("{{convert|20|acre|m2}}", "20 acres (81,000 m2)", "Louisiana_Tech_University"),
        (
            "{{convert|14|m|ft|adj=mid|[[Draft (hull)|draught]]}}",
            "14-metre draught (46 ft)",
            "Uruguay",
        ),  # hyperlink is ignored
        ("{{convert|15|km|mi}}", "15 kilometres (9.3 mi)", "Uruguay"),
        ("{{convert|1200|km|mi|abbr=on}}", "1,200 km (750 mi)", "Uruguay"),
        (
            "{{convert|32.5|mpgus|L/100 km+mpgus+mpgimp|abbr=on|order=out}}",
            "7.2 L/100 km (32.5 mpg‑US; 39.0 mpg‑imp)",
            "Volkswagen_Beetle",
        ),
        (
            "{{convert|130|km/h|mi/h|0|abbr=on}}",
            "130 km/h (81 mph)",
            "Volkswagen_Beetle",
        ),
        (
            "{{Convert|40|L|usgal impgal|0|abbr=on }}",
            "40 L (11 US gal; 9 imp gal)",
            "Volkswagen_Beetle",
        ),
        ("{{convert|50|hp|kW|0|abbr=on}}", "50 hp (37 kW)", "Volkswagen_Beetle"),
        (
            "{{convert|98.1|Nm|lb·ft|abbr=on}}",
            "98.1 N⋅m (72.4 lb⋅ft)",
            "Volkswagen_Beetle",
        ),
        (
            "{{convert |100|km/h|mph|0| abbr=on}}",
            "100 km/h (62 mph)",
            "Volkswagen_Beetle",
        ),
        (
            "{{convert |1800000|mi|km|abbr=on}}",
            "1,800,000 mi (2,900,000 km)",
            "Volkswagen_Beetle",
        ),
        (
            "{{convert |995|cc|cid| abbr =on}}",
            "995 cc (60.7 cu in)",
            "Volkswagen_Beetle",
        ),
        (
            "{{convert|0|–|100|km/h}}",
            "0–100 kilometres per hour (0–62 mph)",
            "Volkswagen_Beetle",
        ),
        (
            "{{convert|6.7|l/100 km}}",
            "6.7 litres per 100 kilometres (42 mpg‑imp; 35 mpg‑US)",
            "Volkswagen_Beetle",
        ),
        (
            "{{cvt|12x14x1|ft|cm|order=flip}}",
            "366 cm × 427 cm × 30 cm (12 ft × 14 ft × 1 ft)",
            "Homo_habilis",
        ),
        (
            "{{cvt|100–120|cm|ftin}}",
            "100–120 cm (3 ft 3 in – 3 ft 11 in)",
            "Homo_habilis",
        ),
        ("{{cvt|20–37|kg}}", "20–37 kg (44–82 lb)", "Homo_habilis"),
        ("{{cvt|245–255|mm}}", "245–255 mm (9.6–10.0 in)", "Homo_habilis"),
        (
            "{{convert|1|e12solar mass|kg|abbr=off|lk=on}}",
            "1 trillion solar masses (2.0×1042 kilograms)",
            "Andromeda_Galaxy",
        ),
        (
            "{{convert|2.5|e6ly|kpc|abbr=off|lk=on}}",
            "2.5 million light-years (770 kiloparsecs)",
            "Andromeda_Galaxy",
        ),
        (
            "{{convert|220,000|ly|kpc|abbr=on|lk=on}}",
            "220,000 ly (67 kpc)",
            "Andromeda_Galaxy",
        ),
        ("{{convert|300|km/s|abbr=on}", "300 km/s (190 mi/s)", "Andromeda_Galaxy"),
        (
            "{{Convert|30|km/s|4 = 2|abbr = on}}",
            "30 km/s (18.64 mi/s)",
            "Michelson–Morley_experiment",
        ),
        (
            "{{Convert| 108000|km/h|abbr = on}}",
            "108,000 km/h (67,000 mph)",
            "Michelson–Morley_experiment",
        ),
        (
            "{{convert|5|ft|spell=in}}",
            "five feet (1.5 m)",
            "Michelson–Morley_experiment",
        ),
        (
            "{{Convert|32|m|adj = on|sp = us}}",
            "32-meter (105 ft)",
            "Michelson–Morley_experiment",
        ),
        (
            "{{convert|62.5|m|ft+royal cubit|abbr=on}}",
            "62.5 m (205 ft; 119.3 cu)",
            "Pyramid_of_Djoser",
        ),
        (
            "{{convert|60-62.5|m|ft+royal cubit|0|abbr=on}}",
            "60–62.5 m (197–205 ft; 115–119 cu)",
            "Pyramid_of_Djoser",
        ),
        ("{{convert|330,400|m3|ft3|abbr=on}}", "330,400 m3 (11,670,000 cu ft)", ""),
        ("{{convert|10|and|20|C|6=0}}", "10 and 20 °C (50 and 68 °F)", "Milan"),
        (
            "{{convert|0|to|9|mph|km/h|0|abbr=on|order=flip}}",
            "0 to 14 km/h (0 to 9 mph)",
            "Milan",
        ),
        ("{{convert|900|acre|km2|adj=mid|-wide}}", "900-acre-wide (3.6 km2)", "Milan"),
        ("{{convert|160|km}}", "160 kilometres (99 mi)", "Milan"),
        (
            "{{convert|3+1//4|to(-)|4+3//4|lb}}",
            "3+1/4 to 4+3/4 pounds (1.5–2.2 kg)",
            "Adze",
        ),
        (
            "{{convert|3|to(-)|4+1//2|inch|round=5}}",
            "3 to 4+1/2 inches (75–115 mm)",
            "Adze",
        ),
        (
            "{{convert|1|to|1+1//3|mi|km}}",
            "1 to 1+1/3 miles (1.6 to 2.1 km)",
            "Romulus,_Michigan",
        ),
        (
            "{{convert|0.35|sqmi|sqkm|2}}",
            "0.35 square miles (0.91 km2)",
            "Romulus,_Michigan",
        ),
        (
            "{{convert|673.7|PD/sqmi|PD/km2|1}}",
            "673.7 inhabitants per square mile (260.1/km2)",
            "Romulus,_Michigan",
        ),
        (
            "{{convert|279.3|/sqmi|/km2|1}}",
            "279.3 per square mile (107.8/km2)",
            "Romulus,_Michigan",
        ),
        (
            "{{convert|1.6|e6sqft|acre_m2}}",
            "1.6 million square feet (37 acres; 150,000 m2)",
            "Stuarts Draft, Virginia",
        ),
        ("{{convert|43|kW|abbr=on}}", "43 kW (58 hp)", "Hydrogen_vehicle"),
        ("{{convert|198|m|ft|0|abbr=on}}", "198 m (650 ft)", "Enercon_E-126"),
        ("{{convert|50|t|lb}}", "50 tonnes (110,000 lb)", ""),
        (
            "{{convert|450,000|t|e6lb|abbr=off}}",
            "450,000 tonnes (990 million pounds)",
            "Wind_turbine",
        ),
        (
            "{{convert|1,500|ST|t|order=flip}}",
            "1,400 tonnes (1,500 short tons)",
            "Wind_turbine",
        ),
        (
            "{{convert|20|to|80|m|ft|sp=us}}",
            "20 to 80 meters (66 to 262 ft)",
            "Wind_turbine",
        ),
    ]
    for template, correct_text, source_article in test_strings:
        text = parse_convert_template(template)
        full_text = text[2]
        if type(full_text) == str:
            full_text = full_text.replace("‐", "‑")
        assert full_text == correct_text


if __name__ == "__main__":
    test_parse_convert_template_no_errors()
    test_parse_convert_template_output_correct()
