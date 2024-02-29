def calculate_wer(original_texts, predicted_texts):
    wer_matrix = []

    for original_text, predicted_text in zip(original_texts, predicted_texts):
        original_text = str(original_text)
        predicted_text = str(predicted_text)

        ref_words = original_text.split()
        hyp_words = predicted_text.split()

      
        substitutions = []
       

        for ref, hyp in zip(ref_words, hyp_words):
            if ref != hyp:
                # Substitutions
                diff_sequence_ref = ''.join(ref_char if ref_char.lower() != hyp_char.lower() else '' for ref_char, hyp_char in zip(ref, hyp))
                diff_sequence_hyp = ''.join(hyp_char if ref_char.lower() != hyp_char.lower() else '' for ref_char, hyp_char in zip(ref, hyp))
                substitutions.append((diff_sequence_ref, diff_sequence_hyp))

            # Deletions
            if len(ref) > len(hyp):
                diff_sequence_ref = ref[len(hyp):] 
                substitutions.append((diff_sequence_ref, None))

            # Insertions
            if len(ref) < len(hyp):
                diff_sequence_ref = hyp[len(ref):]
                substitutions.append((None, diff_sequence_ref))

        wer_matrix.append((substitutions))

    return wer_matrix


