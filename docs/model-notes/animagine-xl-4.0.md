# Animagine XL 4.0 prompt notes

Last updated: 2026-11-10

## Measured prompt behavior

- The local Animagine XL 4.0 pipeline uses two CLIP tokenizers. Both have a 77-token
  context window for the text prompt.
- Count the **final composed prompt**, including the app's enhanced-mode suffix
  (`masterpiece, high score, great score, absurdres`). A 76-token composed prompt
  fit both tokenizers. The 77-token limit includes tokenizer special tokens.
- Long prompts are silently truncated for inference after a warning. Words, tags,
  and punctuation all consume tokens; a comma-separated tag list can be much longer
  than its apparent word count suggests. Keep essential character and pose cues
  early, and verify the final token count rather than counting tags.
- The negative prompt is tokenized separately. In the local check, a 61-token
  composed negative prompt fit; keep its final count within the same 77-token window.

## Observations from Annette tests

- A 168-token prompt dropped most of its late pose and background instructions.
- A 115-token prompt still truncated late pose details. A 91-token composed prompt
  exceeded the limit as well.
- A concise 77-token generation at 1024×1024, 28 steps, CFG 5.5, seed 1180008761
  kept Annette's brown bob and amber eyes recognizable across the batch. Pose
  interpretation varied; some results reclined or knelt instead of forming the
  intended compact furniture shape, and one image duplicated the subject.
- The `warm light skin` wording under warm dramatic lighting still rendered darker
  than the supplied House of Forms references. The next comparison uses `fair skin`
  with neutral studio light. Avoid `tan` for this character unless a darker result
  is wanted.
- The brief original pose tags (`lying on back, legs apart, (pretzel pose), folded,
  frogtie`) produced the most creative variation so far. Detailed anatomical prose
  consumes the token budget and appears to constrain or truncate other cues. Keep
  the pose prompt simple, retain `(pretzel pose)`, and use longer prose as a design
  brief rather than copying all of it into every generation prompt.
- A follow-up four-image batch restored the brief pose tags and used `fair skin` plus
  neutral studio light. The final enhanced prompt measured 60 tokens and the negative
  prompt 61. Hair and eyes stayed close to the references, and skin appeared lighter.
  The pose varied creatively; some images formed a compact pillow-like curl while
  others opened one leg or relaxed the fold. This confirms the simple tag is a good
  exploration prompt, not a precise pose lock.
- Follow-up run: 1024×1024, 28 steps, CFG 5.5, seed 1180008761, four-image batch,
  no input image. Outputs: `outputs/annette/uncategorized/20261110T090742Z_adult_woman_solo_short_wavy_brown_bob_amber_eyes_f742fee4/`.

## Current test recipe

- Character: adult woman, solo, short wavy brown bob, amber eyes, fair skin, glossy skin.
- Clothing: black leotard with gold trim.
- Pose: `lying on back, legs apart, (pretzel pose), folded, frogtie`.
- Background: dark studio, neutral light. The `annette_5` recipe selects this study setup.
- Negative additions: split, legs spread in a V, feet beside face, elongated body,
  broken anatomy, injury, ropes, dark skin, multiple people.
- Keep 1024×1024, 28 steps, CFG 5.5, and the seed fixed while comparing prompt
  wording. The next generation should be checked for token truncation before inference.

The observations above are from a small number of local batches, not a guarantee of
general model behavior. See each run's `metadata.json` for the exact prompt and settings.
