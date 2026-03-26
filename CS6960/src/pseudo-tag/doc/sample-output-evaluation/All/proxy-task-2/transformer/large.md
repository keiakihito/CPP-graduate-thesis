~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main*
.venv ❯ python -m src.evaluation.run_batch_eval \
  data/wav \  
  data/output/embeddings/transformer_large/transformer_large_embeddings.npy \
  data/output/embeddings/transformer_large/transformer_large_metadata.json \  
  data/output/labels/pseudo_labels.csv \  
  --model-name transformer_large \
  --top-k 5 \  
  --relevance-strategy tag_overlap  

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
Top-k results:
1. score=0.960531 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
2. score=0.952006 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
3. score=0.949480 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.949154 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
5. score=0.947385 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [0, 0, 1, 1, 1]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 0.618289

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
Top-k results:
1. score=0.991627 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.988055 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.978771 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.978103 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
5. score=0.975237 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 1, 0, 1, 0]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 0.967468

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Top-k results:
1. score=0.984008 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.979157 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
3. score=0.977738 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
4. score=0.975831 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.974413 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
total_relevant_in_corpus: 6
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 0, 0, 0, 1]
precision@5: 0.200000
recall@5: 0.166667
f1@5: 0.181818
ndcg@5: 0.386853

=== Query: track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav ===
Top-k results:
1. score=0.988741 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.988055 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
3. score=0.984007 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
4. score=0.980464 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.980380 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 1, 0, 0, 1]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 0.946902

=== Query: track_10_Tango_Jalousie.wav ===
Top-k results:
1. score=0.980497 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.980066 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.978103 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.975145 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.971102 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
total_relevant_in_corpus: 6
relevant_in_top_k: 4
top_k: 5
Relevance: [1, 1, 1, 0, 1]
precision@5: 0.800000
recall@5: 0.666667
f1@5: 0.727273
ndcg@5: 0.982892

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
Top-k results:
1. score=0.991627 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
2. score=0.988741 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.981944 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.980497 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
5. score=0.979483 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 1, 0, 1, 0]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 0.967468

=== Query: track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav ===
Top-k results:
1. score=0.979157 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.978698 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
3. score=0.973407 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.972852 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.971902 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
total_relevant_in_corpus: 2
relevant_in_top_k: 0
top_k: 5
Relevance: [0, 0, 0, 0, 0]
precision@5: 0.000000
recall@5: 0.000000
f1@5: 0.000000
ndcg@5: 0.000000

=== Query: track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_24_Simple_Symphony__I__Sentimental_Serenade.wav ===
Top-k results:
1. score=0.978698 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
2. score=0.975801 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.975772 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.973388 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
5. score=0.971741 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
total_relevant_in_corpus: 6
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 0, 0, 1, 0]
precision@5: 0.200000
recall@5: 0.166667
f1@5: 0.181818
ndcg@5: 0.430677

=== Query: track_26_Plink__Plank__Plunk.wav ===
Top-k results:
1. score=0.968499 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
2. score=0.965115 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
3. score=0.964643 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.964242 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
5. score=0.959910 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.693426

=== Query: track_27_Concerto_Grosso.wav ===
Top-k results:
1. score=0.974801 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
2. score=0.971960 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
3. score=0.970975 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
4. score=0.970887 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
5. score=0.966785 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
total_relevant_in_corpus: 8
relevant_in_top_k: 4
top_k: 5
Relevance: [2, 1, 0, 1, 1]
precision@5: 0.800000
recall@5: 0.500000
f1@5: 0.615385
ndcg@5: 0.975196

=== Query: track_30_Passacaglia.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
Top-k results:
1. score=0.980380 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.980066 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
3. score=0.979409 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.977866 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
5. score=0.977844 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 1, 0, 1, 0]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 0.967468

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Top-k results:
1. score=0.989770 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.987801 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.982275 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
4. score=0.976900 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
5. score=0.975237 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.693426

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
Top-k results:
1. score=0.989770 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
2. score=0.989658 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.979483 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.976061 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.974620 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
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
1. score=0.978296 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
2. score=0.976900 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
3. score=0.971947 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.970672 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.962541 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
total_relevant_in_corpus: 2
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 0, 0, 1, 0]
precision@5: 0.200000
recall@5: 0.500000
f1@5: 0.285714
ndcg@5: 0.430677

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Top-k results:
1. score=0.927211 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
2. score=0.923106 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
3. score=0.921566 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
4. score=0.913689 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
5. score=0.906963 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
total_relevant_in_corpus: 8
relevant_in_top_k: 3
top_k: 5
Relevance: [2, 2, 1, 0, 0]
precision@5: 0.600000
recall@5: 0.375000
f1@5: 0.461538
ndcg@5: 1.000000

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
Top-k results:
1. score=0.982275 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
2. score=0.978296 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
3. score=0.974801 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
4. score=0.972611 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
5. score=0.969461 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
total_relevant_in_corpus: 8
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 0, 2, 1, 0]
precision@5: 0.600000
recall@5: 0.375000
f1@5: 0.461538
ndcg@5: 0.709447

=== Query: track_6_Prelude_A_L_Unisson.wav ===
Top-k results:
1. score=0.977738 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.968296 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.966785 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
4. score=0.965066 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
5. score=0.962764 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 0, 1, 0, 1]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.543771

=== Query: track_70_Dedication.wav ===
Top-k results:
1. score=0.966860 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
2. score=0.961054 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
3. score=0.953689 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.950529 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
5. score=0.948032 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
total_relevant_in_corpus: 8
relevant_in_top_k: 4
top_k: 5
Relevance: [2, 2, 1, 1, 0]
precision@5: 0.800000
recall@5: 0.500000
f1@5: 0.615385
ndcg@5: 1.000000

=== Query: track_95_Nocturne.wav ===
Top-k results:
1. score=0.989658 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.987801 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
3. score=0.981944 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
4. score=0.980464 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.979409 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
total_relevant_in_corpus: 8
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 0.125000
f1@5: 0.153846
ndcg@5: 0.630930

=== Batch Evaluation Summary ===
valid_queries: 19
skipped_queries: 3
precision@5: 0.452632
recall@5: 0.361842
f1@5: 0.394868
ndcg@5: 0.681310