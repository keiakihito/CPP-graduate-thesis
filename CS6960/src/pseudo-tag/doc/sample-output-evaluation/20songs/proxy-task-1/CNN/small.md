~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main*
.venv ❯   python -m src.evaluation.run_batch_eval \
    data/wav \
    data/output/embeddings/cnn_small/cnn_small_embeddings.npy \
    data/output/embeddings/cnn_small/cnn_small_metadata.json \
    data/output/labels/pseudo_labels.csv \
    --model-name cnn_small \
    --top-k 5 \
    --relevance-strategy composer

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Top-k results:
1. score=0.990504 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
2. score=0.983007 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.981669 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.980014 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.978051 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
total_relevant_in_corpus: 1
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav ===
Top-k results:
1. score=0.994067 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.990002 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.987313 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.980374 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
5. score=0.980298 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
total_relevant_in_corpus: 1
relevant_in_top_k: 0
top_k: 5
Relevance: [0, 0, 0, 0, 0]
precision@5: 0.000000
recall@5: 0.000000
f1@5: 0.000000
ndcg@5: 0.000000

=== Query: track_10_Tango_Jalousie.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_24_Simple_Symphony__I__Sentimental_Serenade.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_26_Plink__Plank__Plunk.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_27_Concerto_Grosso.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_30_Passacaglia.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Top-k results:
1. score=0.990504 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.986720 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.975679 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.974658 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
5. score=0.974577 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
total_relevant_in_corpus: 1
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.386853

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
Top-k results:
1. score=0.994067 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.992019 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
3. score=0.991628 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.981871 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
5. score=0.980999 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
total_relevant_in_corpus: 1
relevant_in_top_k: 0
top_k: 5
Relevance: [0, 0, 0, 0, 0]
precision@5: 0.000000
recall@5: 0.000000
f1@5: 0.000000
ndcg@5: 0.000000

=== Query: track_43_To_A_Wild_Rose.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
Top-k results:
1. score=0.968008 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.967824 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.967107 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.962961 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
5. score=0.962098 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
total_relevant_in_corpus: 1
relevant_in_top_k: 0
top_k: 5
Relevance: [0, 0, 0, 0, 0]
precision@5: 0.000000
recall@5: 0.000000
f1@5: 0.000000
ndcg@5: 0.000000

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Top-k results:
1. score=0.959662 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
2. score=0.957211 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
3. score=0.950367 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.945168 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
5. score=0.940628 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
total_relevant_in_corpus: 1
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 1.000000
f1@5: 0.333333
ndcg@5: 0.630930

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_6_Prelude_A_L_Unisson.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_70_Dedication.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_95_Nocturne.wav ===
No relevant items for this query. Skipping evaluation.

=== Batch Evaluation Summary ===
valid_queries: 6
skipped_queries: 16
precision@5: 0.100000
recall@5: 0.500000
f1@5: 0.166667
ndcg@5: 0.234106