~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main*
.venv ❯ python -m src.evaluation.run_batch_eval \
    data/wav \
    data/output/embeddings/cnn_small/cnn_small_embeddings.npy \
    data/output/embeddings/cnn_small/cnn_small_metadata.json \
    data/output/labels/pseudo_labels.csv \
    --model-name cnn_small \
    --top-k 5 \
    --relevance-strategy tag_overlap

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
Top-k results:
1. score=0.947218 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.943189 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.937856 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.936825 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
5. score=0.934932 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 1, 0, 1, 0]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 0.967468

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
Top-k results:
1. score=0.992019 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.989955 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.987313 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.981669 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
5. score=0.979375 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.693426

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Top-k results:
1. score=0.990504 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
2. score=0.983007 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.981669 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.980014 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.978051 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
total_relevant_in_corpus: 6
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 0.166667
f1@5: 0.181818
ndcg@5: 0.630930

=== Query: track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav ===
Top-k results:
1. score=0.994067 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.990002 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.987313 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.980374 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
5. score=0.980298 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.693426

=== Query: track_10_Tango_Jalousie.wav ===
Top-k results:
1. score=0.985928 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.968008 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
3. score=0.962682 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.962206 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.959881 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [1, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.877215

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
Top-k results:
1. score=0.991628 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.990002 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.989955 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.980677 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
5. score=0.966772 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.693426

=== Query: track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav ===
Top-k results:
1. score=0.993270 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.975336 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
3. score=0.971434 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.968816 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.957441 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
total_relevant_in_corpus: 2
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 0.500000
f1@5: 0.285714
ndcg@5: 0.386853

=== Query: track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_24_Simple_Symphony__I__Sentimental_Serenade.wav ===
Top-k results:
1. score=0.993270 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
2. score=0.976185 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
3. score=0.974658 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.959850 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.954385 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 0, 1, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.650921

=== Query: track_26_Plink__Plank__Plunk.wav ===
Top-k results:
1. score=0.887583 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
2. score=0.877472 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
3. score=0.851202 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
4. score=0.843314 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
5. score=0.829339 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [1, 0, 0, 0, 1]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.850345

=== Query: track_27_Concerto_Grosso.wav ===
Top-k results:
1. score=0.969938 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.968816 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
3. score=0.968299 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.966181 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
5. score=0.962007 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
total_relevant_in_corpus: 8
relevant_in_top_k: 2
top_k: 5
Relevance: [1, 0, 1, 0, 0]
precision@5: 0.400000
recall@5: 0.250000
f1@5: 0.307692
ndcg@5: 0.919721

=== Query: track_30_Passacaglia.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
Top-k results:
1. score=0.985928 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.970176 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.967824 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
4. score=0.962795 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.962277 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 1, 0, 0, 1]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 0.946902

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Top-k results:
1. score=0.990504 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.986720 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.975679 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.974658 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
5. score=0.974577 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
total_relevant_in_corpus: 6
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 0.166667
f1@5: 0.181818
ndcg@5: 0.630930

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
Top-k results:
1. score=0.994067 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.992019 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
3. score=0.991628 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.981871 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
5. score=0.980999 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
total_relevant_in_corpus: 2
relevant_in_top_k: 0
top_k: 5
Relevance: [0, 0, 0, 0, 0]
precision@5: 0.000000
recall@5: 0.000000
f1@5: 0.000000
ndcg@5: 0.000000

=== Query: track_43_To_A_Wild_Rose.wav ===
Top-k results:
1. score=0.957441 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
2. score=0.954385 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
3. score=0.949527 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.948890 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
5. score=0.946177 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
total_relevant_in_corpus: 2
relevant_in_top_k: 1
top_k: 5
Relevance: [1, 0, 0, 0, 0]
precision@5: 0.200000
recall@5: 0.500000
f1@5: 0.285714
ndcg@5: 1.000000

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Top-k results:
1. score=0.959662 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
2. score=0.957211 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
3. score=0.950367 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.945168 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.940628 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
total_relevant_in_corpus: 8
relevant_in_top_k: 2
top_k: 5
Relevance: [1, 0, 0, 2, 0]
precision@5: 0.400000
recall@5: 0.250000
f1@5: 0.307692
ndcg@5: 0.631251

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
Top-k results:
1. score=0.971243 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
2. score=0.962795 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.960685 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.959791 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
5. score=0.952180 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
total_relevant_in_corpus: 8
relevant_in_top_k: 2
top_k: 5
Relevance: [2, 0, 0, 1, 0]
precision@5: 0.400000
recall@5: 0.250000
f1@5: 0.307692
ndcg@5: 0.944848

=== Query: track_6_Prelude_A_L_Unisson.wav ===
Top-k results:
1. score=0.968171 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
2. score=0.959791 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
3. score=0.959662 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
4. score=0.956055 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
5. score=0.954200 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 1, 1, 0, 0]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 1.000000

=== Query: track_70_Dedication.wav ===
Top-k results:
1. score=0.795531 path=data/wav/track_26_Plink__Plank__Plunk.wav file_id=track_26_Plink__Plank__Plunk
2. score=0.756421 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
3. score=0.746679 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
4. score=0.742133 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.728574 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
total_relevant_in_corpus: 8
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 0, 0, 2, 0]
precision@5: 0.200000
recall@5: 0.125000
f1@5: 0.153846
ndcg@5: 0.430677

=== Query: track_95_Nocturne.wav ===
Top-k results:
1. score=0.986720 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
2. score=0.983007 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
3. score=0.980999 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.980298 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.974468 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
total_relevant_in_corpus: 8
relevant_in_top_k: 2
top_k: 5
Relevance: [1, 1, 0, 0, 0]
precision@5: 0.400000
recall@5: 0.250000
f1@5: 0.307692
ndcg@5: 1.000000

=== Batch Evaluation Summary ===
valid_queries: 19
skipped_queries: 3
precision@5: 0.357895
recall@5: 0.313596
f1@5: 0.323045
ndcg@5: 0.734123