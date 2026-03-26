~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/src/pseudo-tag main*
.venv ❯ python -m src.evaluation.run_batch_eval \
  data/wav \  
  data/output/embeddings/cnn_medium/cnn_medium_embeddings.npy \                
  data/output/embeddings/cnn_medium/cnn_medium_metadata.json \                
  data/output/labels/pseudo_labels.csv \  
  --model-name cnn_medium \        
  --top-k 5 \  
  --relevance-strategy tag_overlap  

=== Query: track_101_Sabre_Dance_from_Gayane_Ballet.wav ===
Top-k results:
1. score=0.973195 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.964358 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.960113 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
4. score=0.951714 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.950956 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 1, 0, 0, 1]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 0.946902

=== Query: track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav ===
Top-k results:
1. score=0.990853 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.989814 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.985891 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.983688 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
5. score=0.983429 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
total_relevant_in_corpus: 6
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 0.166667
f1@5: 0.181818
ndcg@5: 0.630930

=== Query: track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav ===
Top-k results:
1. score=0.993387 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
2. score=0.988013 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.983689 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.980487 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.978748 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
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
1. score=0.991084 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.989047 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
3. score=0.987349 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.978149 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
5. score=0.974769 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 0, 1, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.650921

=== Query: track_10_Tango_Jalousie.wav ===
Top-k results:
1. score=0.991207 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
2. score=0.980717 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
3. score=0.965506 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.961861 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.955270 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [1, 0, 1, 0, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.919721

=== Query: track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav ===
Top-k results:
1. score=0.991680 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.989814 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
3. score=0.989047 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.983595 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.976192 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
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
1. score=0.992554 path=data/wav/track_24_Simple_Symphony__I__Sentimental_Serenade.wav file_id=track_24_Simple_Symphony__I__Sentimental_Serenade
2. score=0.978748 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
3. score=0.969081 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.963710 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.953797 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
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
1. score=0.992554 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
2. score=0.977038 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
3. score=0.971818 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
4. score=0.958067 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.955746 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
total_relevant_in_corpus: 6
relevant_in_top_k: 3
top_k: 5
Relevance: [0, 1, 0, 1, 1]
precision@5: 0.600000
recall@5: 0.500000
f1@5: 0.545455
ndcg@5: 0.679731

=== Query: track_26_Plink__Plank__Plunk.wav ===
Top-k results:
1. score=0.887791 path=data/wav/track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace.wav file_id=track_19_Piano_Concerto_No_1_in_E_Minor__Op__11__III__Rondo__Vivace
2. score=0.884443 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
3. score=0.879597 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
4. score=0.878861 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.872152 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 0, 0, 1]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.624051

=== Query: track_27_Concerto_Grosso.wav ===
Top-k results:
1. score=0.978452 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.973787 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
3. score=0.969601 path=data/wav/track_30_Passacaglia.wav file_id=track_30_Passacaglia
4. score=0.964332 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.963817 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
total_relevant_in_corpus: 8
relevant_in_top_k: 3
top_k: 5
Relevance: [1, 1, 0, 2, 0]
precision@5: 0.600000
recall@5: 0.375000
f1@5: 0.461538
ndcg@5: 0.707579

=== Query: track_30_Passacaglia.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_33_Introduction___Rondo-Capriccioso.wav ===
Top-k results:
1. score=0.991207 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.979936 path=data/wav/track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav file_id=track_69_The_Seasons__Op__37b__October_-__Autumn_Song
3. score=0.964380 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.963989 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.962405 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [1, 0, 1, 0, 0]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.919721

=== Query: track_34_Verklärte_Nacht__Op__4.wav ===
Top-k results:
1. score=0.993387 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.989412 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
3. score=0.983429 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.980960 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
5. score=0.973787 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
total_relevant_in_corpus: 6
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 0, 0, 1]
precision@5: 0.400000
recall@5: 0.333333
f1@5: 0.363636
ndcg@5: 0.624051

=== Query: track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav ===
Top-k results:
1. score=0.991680 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
2. score=0.991084 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
3. score=0.990853 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
4. score=0.990822 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.980960 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
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
1. score=0.955731 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
2. score=0.943980 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
3. score=0.943728 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
4. score=0.942699 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
5. score=0.933130 path=data/wav/track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello.wav file_id=track_14_Serenade_For_Strings_In_G_Minor__Op__27a__IV__Finale__Lento-Presto_Al_Saltarello
total_relevant_in_corpus: 2
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 1, 0, 0, 0]
precision@5: 0.200000
recall@5: 0.500000
f1@5: 0.285714
ndcg@5: 0.630930

=== Query: track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav ===
No relevant items for this query. Skipping evaluation.

=== Query: track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav ===
Top-k results:
1. score=0.965858 path=data/wav/track_6_Prelude_A_L_Unisson.wav file_id=track_6_Prelude_A_L_Unisson
2. score=0.962586 path=data/wav/track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro.wav file_id=track_53_Concerto_In_E_Major__Spring___Op__8_No__1__1__Allegro
3. score=0.962314 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
4. score=0.945217 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
5. score=0.944225 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
total_relevant_in_corpus: 8
relevant_in_top_k: 2
top_k: 5
Relevance: [1, 0, 0, 0, 2]
precision@5: 0.400000
recall@5: 0.250000
f1@5: 0.307692
ndcg@5: 0.595043

=== Query: track_69_The_Seasons__Op__37b__October_-__Autumn_Song.wav ===
Top-k results:
1. score=0.980717 path=data/wav/track_10_Tango_Jalousie.wav file_id=track_10_Tango_Jalousie
2. score=0.979936 path=data/wav/track_33_Introduction___Rondo-Capriccioso.wav file_id=track_33_Introduction___Rondo-Capriccioso
3. score=0.977199 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.973976 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.966255 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
total_relevant_in_corpus: 8
relevant_in_top_k: 1
top_k: 5
Relevance: [0, 0, 2, 0, 0]
precision@5: 0.200000
recall@5: 0.125000
f1@5: 0.153846
ndcg@5: 0.500000

=== Query: track_6_Prelude_A_L_Unisson.wav ===
Top-k results:
1. score=0.966369 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
2. score=0.965858 path=data/wav/track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio.wav file_id=track_57_Concerto_In_G_Minor__Summer___Op__8_No__2__2__Adagio
3. score=0.961308 path=data/wav/track_95_Nocturne.wav file_id=track_95_Nocturne
4. score=0.958370 path=data/wav/track_27_Concerto_Grosso.wav file_id=track_27_Concerto_Grosso
5. score=0.954581 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
total_relevant_in_corpus: 6
relevant_in_top_k: 4
top_k: 5
Relevance: [0, 1, 1, 1, 1]
precision@5: 0.800000
recall@5: 0.666667
f1@5: 0.727273
ndcg@5: 0.760640

=== Query: track_70_Dedication.wav ===
Top-k results:
1. score=0.741145 path=data/wav/track_43_To_A_Wild_Rose.wav file_id=track_43_To_A_Wild_Rose
2. score=0.714273 path=data/wav/track_101_Sabre_Dance_from_Gayane_Ballet.wav file_id=track_101_Sabre_Dance_from_Gayane_Ballet
3. score=0.707851 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
4. score=0.703394 path=data/wav/track_26_Plink__Plank__Plunk.wav file_id=track_26_Plink__Plank__Plunk
5. score=0.695172 path=data/wav/track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio.wav file_id=track_15_Adagio_and_Fuge_in_C_Minor__KV_546__I__Adagio
total_relevant_in_corpus: 8
relevant_in_top_k: 0
top_k: 5
Relevance: [0, 0, 0, 0, 0]
precision@5: 0.000000
recall@5: 0.000000
f1@5: 0.000000
ndcg@5: 0.000000

=== Query: track_95_Nocturne.wav ===
Top-k results:
1. score=0.990822 path=data/wav/track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand.wav file_id=track_38_Minimax__Repertorium_Für_Militärmusik___IV___Löwenzähnchen_An_Baches_Rand
2. score=0.989412 path=data/wav/track_34_Verklärte_Nacht__Op__4.wav file_id=track_34_Verklärte_Nacht__Op__4
3. score=0.988013 path=data/wav/track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo.wav file_id=track_106_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Allegretto_Ma_Non_Troppo
4. score=0.987349 path=data/wav/track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag.wav file_id=track_108_Quartetto_Serioso_No_11_In_F_Minor__Op_95_-_Larghetto_Espressivo_-_Allegretto_Ag
5. score=0.985891 path=data/wav/track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr.wav file_id=track_102_String_Symphony_After_Kreutzer_Sonata_In_A_Major__Op__47_-_Adagio_Sostenuto_-_Pr
total_relevant_in_corpus: 8
relevant_in_top_k: 2
top_k: 5
Relevance: [0, 1, 1, 0, 0]
precision@5: 0.400000
recall@5: 0.250000
f1@5: 0.307692
ndcg@5: 0.693426

=== Batch Evaluation Summary ===
valid_queries: 19
skipped_queries: 3
precision@5: 0.347368
recall@5: 0.289474
f1@5: 0.309480
ndcg@5: 0.589895

