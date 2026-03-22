~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main* 7m 21s
.venv ❯   python -m src.evaluation.run_batch_eval \
  data/wav \
  data/output/embeddings/cnn_large/cnn_large_embeddings.npy \
  data/output/embeddings/cnn_large/cnn_large_metadata.json \
  data/output/labels/pseudo_labels.csv \
  --model-name cnn_large \
  --top-k 5 \
  --relevance-strategy tag_overlap

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
Top-k results:
1. score=0.873446 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.872111 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.869840 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.854831 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.850075 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
Relevance: [1, 1, 0, 0, 1]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 0.946902

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
Top-k results:
1. score=0.880360 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.877955 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
3. score=0.877375 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.875091 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.869791 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [1, 0, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.919721

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Top-k results:
1. score=0.885110 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.878478 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.877050 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.876773 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
5. score=0.876705 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
Relevance: [1, 1, 0, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 1.000000

=== Query: track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav ===
Top-k results:
1. score=0.896606 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.893695 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.882789 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.881122 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
5. score=0.880550 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.693426

=== Query: track_10_Tango_Jalousie.wav ===
Top-k results:
1. score=0.929118 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.909886 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
3. score=0.894944 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
4. score=0.867991 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.863825 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
Top-k results:
1. score=0.874347 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.872332 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
3. score=0.868976 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.863291 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
5. score=0.861155 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
Relevance: [1, 0, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.919721

=== Query: track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_24_Simple_Symphony__I__Sentimental_Serenade.wav ===
Top-k results:
1. score=0.901993 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
2. score=0.893392 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
3. score=0.885429 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.881687 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
5. score=0.879381 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [0, 1, 0, 0, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.624051

=== Query: track_26_Plink__Plank__Plunk.wav ===
Top-k results:
1. score=0.794244 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
2. score=0.787820 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.782602 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.776764 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.770225 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [1, 1, 0, 0, 1]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 0.946902

=== Query: track_27_Concerto_Grosso.wav ===
Top-k results:
1. score=0.866679 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
2. score=0.828022 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
3. score=0.812693 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
4. score=0.812204 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.807241 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
Relevance: [0, 0, 1, 0, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.543771

=== Query: track_30_Passacaglia.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
Top-k results:
1. score=0.896824 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.873674 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
3. score=0.868693 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
4. score=0.846869 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
5. score=0.837422 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
Relevance: [1, 0, 0, 0, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.850345

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Top-k results:
1. score=0.837959 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
2. score=0.834827 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
3. score=0.834159 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.831909 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.830051 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_43_To_A_Wild_Rose.wav ===
Top-k results:
1. score=0.892134 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.891846 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
3. score=0.891093 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
4. score=0.889371 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.882589 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
Relevance: [0, 0, 0, 1, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.501266

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Top-k results:
1. score=0.945282 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.926395 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
3. score=0.923299 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
4. score=0.913781 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
5. score=0.906548 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
Relevance: [0, 0, 0, 1, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.501266

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
Top-k results:
1. score=0.915609 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
2. score=0.911009 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.908376 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
4. score=0.895857 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
5. score=0.895544 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Query: track_6_Prelude_A_L_Unisson.wav ===
Top-k results:
1. score=0.867994 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.866758 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
3. score=0.860434 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
4. score=0.857265 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.856343 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
Relevance: [0, 1, 0, 0, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.624051

=== Query: track_70_Dedication.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_95_Nocturne.wav ===
Top-k results:
1. score=0.870707 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.870016 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.869977 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.869695 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
5. score=0.865889 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Batch Evaluation Summary ===
valid_queries: 16
skipped_queries: 6
precision@5: 0.425000
recall@5: 1.000000
f1@5: 0.593750
ndcg@5: 0.786268