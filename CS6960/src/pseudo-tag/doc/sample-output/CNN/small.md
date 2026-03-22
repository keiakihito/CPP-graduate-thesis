~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main* 10s
.venv ❯ python -m src.evaluation.run_batch_eval \
  data/wav \
  data/output/embeddings/transformer_small_embeddings.npy \
  data/output/embeddings/transformer_small_metadata.json \
  data/output/labels/pseudo_labels.csv \
  --model-name transformer_small \
  --top-k 5 \
  --relevance-strategy tag_overlap

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag/.venv/lib/python3.11/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.2.0)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
No relevant items for this query. Skipping evaluation.

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
Top-k results:
1. score=0.948523 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.942172 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.941280 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
4. score=0.936989 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
5. score=0.936479 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Top-k results:
1. score=0.930136 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
2. score=0.897418 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
3. score=0.893988 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.890002 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.879649 path=data/wav/track_70_Dedication.wav file_id=track_70_Dedication
Relevance: [0, 0, 0, 1, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.501266

=== Query: track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav ===
Top-k results:
1. score=0.957184 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.943487 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
3. score=0.940647 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
4. score=0.938732 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.934430 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_10_Tango_Jalousie.wav ===
Top-k results:
1. score=0.969073 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
2. score=0.940906 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
3. score=0.940032 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.933550 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.931648 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
Relevance: [0, 0, 1, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.500000

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
Top-k results:
1. score=0.963332 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.959119 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.945087 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.942172 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
5. score=0.941024 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
Relevance: [0, 0, 0, 1, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.430677

=== Query: track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav ===
Top-k results:
1. score=0.934800 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
2. score=0.929905 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
3. score=0.925460 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
4. score=0.921768 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.914587 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
Relevance: [0, 0, 0, 1, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.430677

=== Query: track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_24_Simple_Symphony__I__Sentimental_Serenade.wav ===
Top-k results:
1. score=0.968027 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
2. score=0.963332 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.957932 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.947853 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.944587 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 0, 1, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.570642

=== Query: track_26_Plink__Plank__Plunk.wav ===
Top-k results:
1. score=0.889599 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
2. score=0.877536 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.871312 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.869048 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
5. score=0.865907 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 1, 1, 1, 0]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 0.732829

=== Query: track_27_Concerto_Grosso.wav ===
Top-k results:
1. score=0.905249 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
2. score=0.903118 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
3. score=0.903019 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.899474 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
5. score=0.895348 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
Relevance: [0, 0, 1, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.500000

=== Query: track_30_Passacaglia.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
Top-k results:
1. score=0.946800 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
2. score=0.940032 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.935718 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.934022 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
5. score=0.933573 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Top-k results:
1. score=0.903019 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
2. score=0.893107 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.883560 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
4. score=0.876575 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
5. score=0.872728 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
Top-k results:
1. score=0.960166 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
2. score=0.958669 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.948523 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.947358 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.944587 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_43_To_A_Wild_Rose.wav ===
Top-k results:
1. score=0.968027 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.964155 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.962593 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.959119 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.958669 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Top-k results:
1. score=0.962975 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
2. score=0.940906 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.934022 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.931710 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.927510 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
Top-k results:
1. score=0.970016 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
2. score=0.962593 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.957932 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
4. score=0.947358 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.945087 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [1, 0, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.919721

=== Query: track_6_Prelude_A_L_Unisson.wav ===
Top-k results:
1. score=0.899902 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
2. score=0.895953 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
3. score=0.895933 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
4. score=0.889366 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.889157 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
Relevance: [1, 0, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 1.000000

=== Query: track_70_Dedication.wav ===
Top-k results:
1. score=0.856428 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
2. score=0.833917 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
3. score=0.828147 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
4. score=0.823283 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
5. score=0.823216 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.693426

=== Query: track_95_Nocturne.wav ===
Top-k results:
1. score=0.970016 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
2. score=0.964155 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.960166 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.947853 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
5. score=0.938812 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Batch Evaluation Summary ===
valid_queries: 18
skipped_queries: 4
precision@5: 0.288889
recall@5: 1.000000
ndcg@5: 0.629505

~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main* 30s
.venv ❯ python -m src.evaluation.run_batch_eval \
  data/wav \
  data/output/embeddings/transformer_small_embeddings.npy \
  data/output/embeddings/transformer_small_metadata.json \
  data/output/labels/pseudo_labels.csv \
  --model-name transformer_small \
  --top-k 5 \
  --relevance-strategy tag_overlap

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
/Users/keita-katsumi/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag/.venv/lib/python3.11/site-packages/requests/__init__.py:113: RequestsDependencyWarning: urllib3 (2.6.3) or chardet (7.2.0)/charset_normalizer (3.4.6) doesn't match a supported version!
  warnings.warn(
No relevant items for this query. Skipping evaluation.

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
Top-k results:
1. score=0.948523 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.942172 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.941280 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
4. score=0.936989 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
5. score=0.936479 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Top-k results:
1. score=0.930136 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
2. score=0.897418 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
3. score=0.893988 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.890002 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.879649 path=data/wav/track_70_Dedication.wav file_id=track_70_Dedication
Relevance: [0, 0, 0, 1, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.501266

=== Query: track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav ===
Top-k results:
1. score=0.957184 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.943487 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
3. score=0.940647 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
4. score=0.938732 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.934430 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_10_Tango_Jalousie.wav ===
Top-k results:
1. score=0.969073 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
2. score=0.940906 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
3. score=0.940032 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.933550 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.931648 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
Relevance: [0, 0, 1, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.500000

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
Top-k results:
1. score=0.963332 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.959119 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.945087 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.942172 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
5. score=0.941024 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
Relevance: [0, 0, 0, 1, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.430677

=== Query: track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav ===
Top-k results:
1. score=0.934800 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
2. score=0.929905 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
3. score=0.925460 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
4. score=0.921768 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.914587 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
Relevance: [0, 0, 0, 1, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.430677

=== Query: track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_24_Simple_Symphony__I__Sentimental_Serenade.wav ===
Top-k results:
1. score=0.968027 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
2. score=0.963332 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.957932 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.947853 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.944587 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 0, 1, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.570642

=== Query: track_26_Plink__Plank__Plunk.wav ===
Top-k results:
1. score=0.889599 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
2. score=0.877536 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.871312 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.869048 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
5. score=0.865907 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 1, 1, 1, 0]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 0.732829

=== Query: track_27_Concerto_Grosso.wav ===
Top-k results:
1. score=0.905249 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
2. score=0.903118 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
3. score=0.903019 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.899474 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
5. score=0.895348 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
Relevance: [0, 0, 1, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.500000

=== Query: track_30_Passacaglia.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
Top-k results:
1. score=0.946800 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
2. score=0.940032 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.935718 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.934022 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
5. score=0.933573 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Top-k results:
1. score=0.903019 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
2. score=0.893107 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.883560 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
4. score=0.876575 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
5. score=0.872728 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
Top-k results:
1. score=0.960166 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
2. score=0.958669 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.948523 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.947358 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.944587 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_43_To_A_Wild_Rose.wav ===
Top-k results:
1. score=0.968027 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.964155 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.962593 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.959119 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.958669 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Top-k results:
1. score=0.962975 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
2. score=0.940906 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.934022 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.931710 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.927510 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
Top-k results:
1. score=0.970016 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
2. score=0.962593 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.957932 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
4. score=0.947358 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.945087 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [1, 0, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.919721

=== Query: track_6_Prelude_A_L_Unisson.wav ===
Top-k results:
1. score=0.899902 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
2. score=0.895953 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
3. score=0.895933 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
4. score=0.889366 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.889157 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
Relevance: [1, 0, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 1.000000

=== Query: track_70_Dedication.wav ===
Top-k results:
1. score=0.856428 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
2. score=0.833917 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
3. score=0.828147 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
4. score=0.823283 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
5. score=0.823216 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.693426

=== Query: track_95_Nocturne.wav ===
Top-k results:
1. score=0.970016 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
2. score=0.964155 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.960166 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.947853 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
5. score=0.938812 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Batch Evaluation Summary ===
valid_queries: 18
skipped_queries: 4
precision@5: 0.288889
recall@5: 1.000000
f1@5: 0.435847
ndcg@5: 0.629505

~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main* 30s
.venv ❯     python -m src.evaluation.run_batch_eval \   
    data/wav \
    data/output/embeddings/cnn_small_embeddings.npy \
    data/output/embeddings/cnn_small_metadata.json \
    data/output/labels/pseudo_labels.csv \
    --model-name cnn_small \
    --top-k 5 \
    --relevance-strategy tag_overlap

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.934985 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.931557 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
3. score=0.927265 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.907914 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
5. score=0.895669 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.877215

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.965591 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.957084 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.945362 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.935839 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.934000 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [1, 0, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 1.000000

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.969751 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
2. score=0.937300 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.937168 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.935259 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
5. score=0.933311 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
Relevance: [0, 1, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.650921

=== Query: track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.971373 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.965447 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.955905 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.952089 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
5. score=0.940169 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
Relevance: [0, 1, 0, 1, 1]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 0.679731

=== Query: track_10_Tango_Jalousie.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.968023 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.963012 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
3. score=0.936990 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.927223 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
5. score=0.901772 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
Relevance: [1, 0, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 1.000000

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.965591 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
2. score=0.965192 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.952457 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.951568 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.945477 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [1, 0, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 1.000000

=== Query: track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.964645 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.928537 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.922391 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
4. score=0.922207 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
5. score=0.914899 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
No relevant items for this query. Skipping evaluation.

=== Query: track_24_Simple_Symphony__I__Sentimental_Serenade.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.964645 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
2. score=0.950186 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
3. score=0.945477 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.943105 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.943029 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
Relevance: [0, 0, 0, 1, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.430677

=== Query: track_26_Plink__Plank__Plunk.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.844035 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
2. score=0.827966 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.826473 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.821798 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
5. score=0.820740 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
Relevance: [1, 1, 1, 0, 0]
precision@5: 0.600000
recall@5: 1.000000
f1@5: 0.750000
ndcg@5: 1.000000

=== Query: track_27_Concerto_Grosso.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.948577 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
2. score=0.925859 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
3. score=0.915323 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
4. score=0.909184 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.904098 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
Relevance: [0, 1, 0, 0, 1]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.624051

=== Query: track_30_Passacaglia.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.968023 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.952255 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
3. score=0.927720 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.922497 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
5. score=0.916712 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
Relevance: [1, 0, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 1.000000

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.959589 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
2. score=0.952457 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.948323 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
4. score=0.945362 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
5. score=0.943029 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
Relevance: [1, 0, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 1.000000

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.951568 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.936776 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.935839 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.935532 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.931557 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_43_To_A_Wild_Rose.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.936776 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.915592 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
3. score=0.914914 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.906216 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.902629 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
Relevance: [1, 0, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 1.000000

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
No relevant items for this query. Skipping evaluation.

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.938781 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
2. score=0.927223 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.922497 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.921935 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
5. score=0.901905 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.936990 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.927720 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.926416 path=data/wav/track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto.wav file_id=track_7_Concerto_For_2_Violins_And_Strings__Op__77__I__Allegro_Risoluto
4. score=0.923954 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.920831 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
Relevance: [0, 0, 0, 1, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.430677

=== Query: track_6_Prelude_A_L_Unisson.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.950186 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.927180 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
3. score=0.926151 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.923260 path=data/wav/track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino.wav file_id=track_8_Concerto_For_2_Violins_And_Strings__Op__77__II__Andantino
5. score=0.922391 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_70_Dedication.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
No relevant items for this query. Skipping evaluation.

=== Query: track_95_Nocturne.wav ===
Checkpoint path: /Users/keita-katsumi/panns_data/Cnn14_mAP=0.431.pth
Using CPU.
Top-k results:
1. score=0.965192 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.959589 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
3. score=0.957084 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.943105 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
5. score=0.940286 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
Relevance: [0, 1, 0, 1, 0]
precision@5: 0.400000
recall@5: 1.000000
f1@5: 0.571429
ndcg@5: 0.650921

=== Batch Evaluation Summary ===
valid_queries: 18
skipped_queries: 4
precision@5: 0.288889
recall@5: 1.000000
f1@5: 0.432540
ndcg@5: 0.743320

