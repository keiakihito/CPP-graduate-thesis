~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main*
.venv ❯   python -m src.evaluation.run_batch_eval \
    data/wav \
    data/output/embeddings/cnn_small/cnn_small_embeddings.npy \
    data/output/embeddings/cnn_small/cnn_small_metadata.json \
    data/output/labels/pseudo_labels.csv \
    --model-name cnn_small \
    --top-k 5 \
    --relevance-strategy tag_overlap

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
Top-k results:
1. score=0.892549 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.888456 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.884398 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
4. score=0.883514 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.875811 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [1, 1, 1, 1, 0]
precision@5: 0.800000
recall@5: 1.000000
f1@5: 0.888889
ndcg@5: 1.000000

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
Top-k results:
1. score=0.931774 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.930368 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.923397 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.921249 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.915256 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
Relevance: [1, 1, 1, 0, 1]
precision@5: 0.800000
recall@5: 1.000000
f1@5: 0.888889
ndcg@5: 0.982892

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Top-k results:
1. score=0.951658 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.944694 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.940410 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.937970 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.936505 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav ===
Top-k results:
1. score=0.932728 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.932201 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.923213 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.913445 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
5. score=0.910968 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
Relevance: [1, 1, 0, 1, 0]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 0.967468

=== Query: track_10_Tango_Jalousie.wav ===
Top-k results:
1. score=0.934335 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.896658 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.892348 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.887491 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.885495 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [1, 1, 1, 0, 0]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 1.000000

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
Top-k results:
1. score=0.917916 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.913140 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.909197 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.897571 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
5. score=0.896801 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
Relevance: [1, 1, 0, 1, 1]
precision@5: 0.800000
recall@5: 1.000000
f1@5: 0.888889
ndcg@5: 0.955830

=== Query: track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav ===
Top-k results:
1. score=0.955914 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.951221 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
3. score=0.950326 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.950205 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.948422 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
Relevance: [0, 0, 1, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.500000

=== Query: track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_24_Simple_Symphony__I__Sentimental_Serenade.wav ===
Top-k results:
1. score=0.957117 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.952653 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.951140 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.949505 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.947639 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
Relevance: [1, 1, 0, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 1.000000

=== Query: track_26_Plink__Plank__Plunk.wav ===
Top-k results:
1. score=0.822196 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
2. score=0.816493 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
3. score=0.766901 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
4. score=0.760784 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.760054 path=data/wav/track_70_Dedication.wav file_id=track_70_Dedication
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Query: track_27_Concerto_Grosso.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_30_Passacaglia.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
Top-k results:
1. score=0.902455 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.886179 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.883792 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.872559 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.872412 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [1, 1, 1, 0, 0]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 1.000000

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Top-k results:
1. score=0.913406 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.908755 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
3. score=0.908035 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.904991 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.904491 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_43_To_A_Wild_Rose.wav ===
Top-k results:
1. score=0.941729 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.933022 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
3. score=0.925836 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.925228 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.924589 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Top-k results:
1. score=0.941990 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.939126 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
3. score=0.933358 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.928589 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
5. score=0.920226 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.693426

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
Top-k results:
1. score=0.947165 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.927028 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
3. score=0.926930 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
4. score=0.924346 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.923248 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [0, 1, 0, 0, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.624051

=== Query: track_6_Prelude_A_L_Unisson.wav ===
Top-k results:
1. score=0.930239 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.919158 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
3. score=0.915797 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.908677 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.907033 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [0, 1, 1, 0, 1]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 0.712263

=== Query: track_70_Dedication.wav ===
Top-k results:
1. score=0.797183 path=data/wav/track_26_Plink__Plank__Plunk.wav file_id=track_26_Plink__Plank__Plunk
2. score=0.761275 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
3. score=0.712388 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
4. score=0.705787 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
5. score=0.693242 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_95_Nocturne.wav ===
No relevant items for this query. Skipping evaluation.

=== Batch Evaluation Summary ===
valid_queries: 16
skipped_queries: 6
precision@5: 0.462500
recall@5: 1.000000
f1@5: 0.601190
ndcg@5: 0.756540

