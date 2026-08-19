SYNTHETIC LEGAL CORPUS — for RAG retrieval testing
==================================================
All content is invented. No real people/companies/matters. Based
off my deployed RAGs system for a law firm. 

WHY THIS SET IS SHAPED THIS WAY (the hard cases to test against):

1. NEAR-DUPLICATE indemnification agreements differ ONLY by party name.
   Test: "Which indemnitees' agreements say they relinquished control of the
   shares?" Correct answer = Margaret Ellison, David Okafor, Thomas Vance.
   (Priya Raman and Susan Cho have NO carve-out.) A naive semantic search will
   see all five as nearly identical — retrieval must use party metadata to
   distinguish them. THIS is the core precision test.

2. KEYWORD vs CONCEPT: the phrase "relinquish control" appears in the merger
   agreement (stockholders relinquish control) in a DIFFERENT sense than the
   indemnification carve-outs. Tests whether the system conflates a keyword
   match with the actual concept the user meant.

3. DECOYS: the NDA and lease are corporate documents that mention "change in
   control" or confidentiality but are NOT responsive to a relinquish-control
   query. Good retrieval should rank them LOW.

answer_key.json = ground truth for scoring retrieval precision/recall.

SUGGESTED TEST QUERIES:
  - "Which parties relinquished control of their shares?"
  - "Find the change-in-control carve-out for David Okafor."
  - "Does the Vantage Analytics NDA contain a change of control provision?"
    (answer: no, only lease/employment do; the query names its target because
    the corpus also holds 500+ real contracts with many NDAs among them)
  - "Which documents involve the Coastal Devices merger?"
