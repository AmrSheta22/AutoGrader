import pandas as pd 
import pkg_resources
from symspellpy import SymSpell, Verbosity
import string
import re
import warnings
warnings.filterwarnings("ignore")

class SpellChecker():
    def __init__(self):
        """
        Initialize the spell checker class which is used to correct spelling mistakes in the text with punctuation.
        which is normally removed by the symspellpy library.

        This class uses the symspellpy library to correct spelling mistakes in the text, 
        and it is based on the SymSpell algorithm developed by Wolf Garbe, 
        The below code is a wrapper around the symspellpy library to make it easier to use.

        function lookup: This function is used to correct the spelling of a single word.
        function lookup_compound: This function is used to correct the spelling of a sentence.
        function correct: This function is used to correct the spelling of a sentence with punctuation.
        """
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    def lookup(self, input_term, max_edit_distance=2):
        suggestions = self.sym_spell.lookup(input_term, Verbosity.CLOSEST, max_edit_distance=max_edit_distance,
                                            transfer_casing=True, include_unknown=True)
        return suggestions[0].term

    def lookup_compound(self, input_term, max_edit_distance=2, **kwargs):
        suggestions = self.sym_spell.lookup_compound((input_term), max_edit_distance=max_edit_distance,
                                                     transfer_casing=True, ignore_non_words=True, **kwargs)
        return suggestions[0].term if len(suggestions) > 0 else input_term

    def correct(self, text, **kwargs):
        # correct the spelling of the text with punctuation
        result = ""
        start = 0
        changed = True
        for match in re.finditer(f"[{re.escape(string.punctuation)}]", text):
            end = match.start(0)
            spaces = re.search(r"^(\s+)", text[start: end])
            corrected_text = self.lookup_compound(text[start: end], **kwargs)
            corrected_text = spaces.group(0) + corrected_text if spaces is not None else corrected_text
            spaces = re.search(r"(\s+)$", text[start: end])
            corrected_text = corrected_text + spaces.group(0) if spaces is not None else corrected_text
            corrected_text += match.group(0)
            result = ''.join([result, corrected_text])
            start = match.end(0)
        spaces = re.search(r"^(\s+)", text[start:])
        corrected_text = self.lookup_compound(text[start:], **kwargs)
        corrected_text = spaces.group(0) + corrected_text if spaces is not None else corrected_text
        result = ''.join([result, corrected_text])
        if corrected_text.lower() == text.lower():
            changed = False
        return result
    
def correct_text(text, spell_checker = SpellChecker()):
    try:
        return spell_checker.correct(text)
    except:
        return text