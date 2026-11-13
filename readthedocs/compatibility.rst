Compatibility and migration
===========================

matchms 0.x and 1.0
------------------

The compatibility release supports both API generations without changing the
Word2Vec model format or Spec2Vec's similarity mathematics.

.. list-table::
   :header-rows: 1

   * - Topic
     - matchms 0.x
     - matchms 1.0
   * - ``Spec2Vec.matrix()``
     - Legacy dense NumPy or sparse output
     - ``Scores`` container
   * - ``calculate_scores()``
     - Legacy score entries / ``scores_by_query()``
     - ``scores.to_array("score")``
   * - Top matches
     - ``scores_by_query(query, sort=True)``
     - Sort the relevant matrix column

A score matrix has *reference rows* and *query columns*. Use the first query
column with ``scores.to_array("score")[:, 0]`` to rank its references.

Model compatibility
-------------------

Gensim remains the implementation of Word2Vec for this release. Model
serialization remains Gensim-based. The eventual replacement of Gensim is a
separate project and is not part of this API-compatibility update.

Python support
--------------

This release line targets Python 3.10--3.13. Python 3.7 and 3.8 instructions
from older documentation are obsolete.

Testing prereleases
-------------------

The compatibility CI explicitly tests ``matchms==1.0.0.dev1`` in addition to
stable matchms 0.x. A normal dependency resolution can prefer stable releases;
install the desired prerelease explicitly if you want to reproduce that job.
