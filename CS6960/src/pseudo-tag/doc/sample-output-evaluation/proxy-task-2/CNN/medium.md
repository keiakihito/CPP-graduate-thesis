~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main*
.venv ❯   python -m src.evaluation.run_batch_eval \
  data/wav \
  data/output/embeddings/cnn_medium/cnn_medium_embeddings.npy \
  data/output/embeddings/cnn_medium/cnn_medium_metadata.json \
  data/output/labels/pseudo_labels.csv \
  --model-name cnn_medium \
  --top-k 5 \
  --relevance-strategy tag_overlap

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
Top-k results:
1. score=0.858574 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.857646 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.850641 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.843425 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
5. score=0.842275 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [1, 1, 1, 1, 0]
precision@5: 0.800000
recall@5: 1.000000
f1@5: 0.888889
ndcg@5: 1.000000

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
Top-k results:
1. score=0.920419 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.920243 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.918842 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
4. score=0.916076 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.914919 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
Relevance: [1, 1, 1, 0, 1]
precision@5: 0.800000
recall@5: 1.000000
f1@5: 0.888889
ndcg@5: 0.982892

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Top-k results:
1. score=0.937306 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.936571 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
3. score=0.928531 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.927939 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.926371 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.693426

=== Query: track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav ===
Top-k results:
1. score=0.901837 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
2. score=0.882278 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.877612 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.877460 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.876293 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [1, 1, 1, 0, 1]
precision@5: 0.800000
recall@5: 1.000000
f1@5: 0.888889
ndcg@5: 0.982892

=== Query: track_10_Tango_Jalousie.wav ===
Top-k results:
1. score=0.918061 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.897318 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
3. score=0.890488 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
4. score=0.882511 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.872854 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [1, 0, 1, 1, 1]
precision@5: 0.800000
recall@5: 1.000000
f1@5: 0.888889
ndcg@5: 0.904717

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
Top-k results:
1. score=0.898041 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
2. score=0.889376 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.882069 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
4. score=0.878823 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
5. score=0.878495 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [1, 1, 1, 1, 0]
precision@5: 0.800000
recall@5: 1.000000
f1@5: 0.888889
ndcg@5: 1.000000

=== Query: track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_24_Simple_Symphony__I__Sentimental_Serenade.wav ===
Top-k results:
1. score=0.940382 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
2. score=0.935864 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
3. score=0.934952 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
4. score=0.934880 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.930694 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
Relevance: [0, 0, 1, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.570642

=== Query: track_26_Plink__Plank__Plunk.wav ===
Top-k results:
1. score=0.793222 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
2. score=0.786943 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
3. score=0.763878 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
4. score=0.761488 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
5. score=0.760914 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
Relevance: [1, 0, 0, 0, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.850345

=== Query: track_27_Concerto_Grosso.wav ===
Top-k results:
1. score=0.867206 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
2. score=0.859816 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.852892 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
4. score=0.846264 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.842869 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_30_Passacaglia.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
Top-k results:
1. score=0.868465 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.851714 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
3. score=0.844330 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
4. score=0.840105 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.826600 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [1, 0, 1, 1, 1]
precision@5: 0.800000
recall@5: 1.000000
f1@5: 0.888889
ndcg@5: 0.904717

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Top-k results:
1. score=0.857121 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.855117 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
3. score=0.851933 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
4. score=0.849136 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.848462 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
Relevance: [0, 1, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.650921

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_43_To_A_Wild_Rose.wav ===
Top-k results:
1. score=0.902819 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
2. score=0.897965 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.881411 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.877421 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.869992 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [0, 0, 1, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.500000

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Top-k results:
1. score=0.951244 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.930842 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
3. score=0.930380 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
4. score=0.928323 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
5. score=0.917172 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
Relevance: [0, 0, 1, 0, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.543771

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
Top-k results:
1. score=0.928883 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.920700 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.904725 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.892654 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
5. score=0.891471 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_6_Prelude_A_L_Unisson.wav ===
Top-k results:
1. score=0.897665 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
2. score=0.894232 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.866729 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.865945 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
5. score=0.864616 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
Relevance: [1, 0, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.919721

=== Query: track_70_Dedication.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_95_Nocturne.wav ===
Top-k results:
1. score=0.909005 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
2. score=0.902091 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.899759 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
4. score=0.896497 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.890961 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 0, 0, 1, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.430677

=== Batch Evaluation Summary ===
valid_queries: 16
skipped_queries: 6
precision@5: 0.500000
recall@5: 1.000000
f1@5: 0.630952
ndcg@5: 0.731777

