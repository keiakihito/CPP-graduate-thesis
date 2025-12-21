# Final Polishing Steps (Future Work)
While the proposal is ready, here are three sophisticated points to keep in mind for the actual thesis writing or defense. You do not need to rewrite the proposal now, but be ready to answer these if a professor asks:

## A. The "Synthetic Sequence" Challenge (Crucial for BERT4Rec)
-  The Issue: You plan to use BERT4Rec, which learns sequential patterns (e.g., Track A → Track B → Track C).
- The Risk: If your "Mock Data" (Schedl) creates users who just pick random songs from a specific genre cluster, there is no sequence for BERT4Rec to learn.
- The Polish: In your actual thesis execution, your mock data generation must mimic transitions. For example, simulate a user starting with a slow tempo track and gradually moving to faster tracks (Tempo Ramp), or moving between acoustically similar keys (Harmonic Mixing). This ensures BERT4Rec has actual patterns to detect.

## B. Defining "Cold Start"
-  The Issue: The proposal mentions the "Cold Start" problem.
-  The Polish: Be precise. You are solving the New Item Cold Start problem (adding new iPalpiti tracks that no one has listened to yet). Content-based audio embeddings (Tamm et al.) are perfect for this. You are not necessarily solving the New User Cold Start problem (a brand new user with no history), as that usually requires demographic data (like Schedl's country data) or onboarding questionnaires.


# Elaboration 
The "Final Polishing Steps" I mentioned are implementation advice for when you actually start building the system after your proposal is approved. These are points to keep in mind so you can defend your choices if a committee member asks specific technical questions during your presentation.
Here is an elaboration on exactly what those steps mean for your research execution:

## 1. The "Synthetic Sequence" Challenge (Crucial for BERT4Rec)
What you need to understand: Your proposal plans to use BERT4Rec, a model designed to predict the next item in a sequence (e.g., "User listened to A, then B, then C... what is D?"). However, since iPalpiti has no real user history, you are generating "Mock Data" (synthetic users) based on Schedl et al.'s archetypes

- The specific challenge: If you generate mock data randomly (e.g., "Cluster 0 User picks Random Song A, then Random Song B"), there is no logical "flow" or transition for BERT4Rec to learn. Real users rarely jump from a slow, sad Adagio to a fast, aggressive Presto and back again randomly.
Implementation Advice (How to fix this when coding): When you write the code to generate your mock user sessions, do not just pick random songs from a cluster. You should simulate Transitions:
- Tempo Ramping: Create synthetic sessions where the songs gradually increase in tempo (e.g., start with a slow violin piece, end with a fast orchestral finale).
-  Harmonic Mixing: Order the synthetic playlist so that adjacent tracks have compatible keys (using the Chroma features mentioned in Shi
).
- Why this matters: This gives BERT4Rec a pattern to learn. If the data is totally random, BERT4Rec will fail to converge, and your results will look bad.

## 2. Precise Definition of "Cold Start"
What you need to understand: Committee members often ask for clarification on "Cold Start" because there are three types. You need to be precise about which one you are solving to avoid confusion.
The specific distinction:
-  New User Cold Start: A new person joins the app. We know nothing about them. (Solved by onboarding questions like "Do you like Violin?", which relates to Schedl’s archetypes
).
-  New Item Cold Start: A new song is uploaded to the archive. No one has listened to it yet. (Solved by Tamm et al.'s audio embeddings
, because the model can "hear" the song and know it sounds like other popular songs).
- System Cold Start: The app is brand new and has no data at all. (This is your specific case with iPalpiti).
Implementation Advice (How to defend this): If asked, state clearly: "This thesis primarily addresses the System Cold Start and New Item Cold Start problems. By using Tamm et al.'s pretrained audio embeddings
, the system effectively 'hears' every song immediately upon upload. It does not need to wait for user interactions to understand that a specific recording of Tchaikovsky is similar to another recording, because the content embeddings already capture that similarity."

## 3. Handling "Repeated Consumption" (Abbattista et al.)
What you need to understand: You added Abbattista et al.
 to your bibliography. Their key finding is that complex models (like BERT4Rec) often fail because they try to predict new songs constantly, whereas real music listeners love listening to the same songs over and over (Repeated Consumption).
Implementation Advice (When building the model):
• Don't force novelty: When you evaluate your model (Phase 3), do not penalize it if it recommends a song the user has already heard.
• The "Personalized Popularity" Feature: Abbattista suggests feeding a "Personalized Popularity" score (how often this specific user listens to this specific artist) into the model
. Since you don't have real history, you can simulate this in your mock data by making your "Synthetic Users" have "favorite songs" that appear multiple times in their listening queues. This makes your synthetic data look more realistic and helps BERT4Rec learn better.

## Summary of Action Items
1. Submit Draft v4.2. It is academically rigorous and correctly formatted.
2. No new papers needed. You have a strong mix of foundational tech (MusiCNN, BERT4Rec) and domain-specific validation (Shi for Classical, Schedl for User Archetypes).
3. Prepare for the defense/implementation: Keep the points above in mind. They show you understand not just what the models are, but how they behave in real-world scenarios.