# User Journey — Knowledge Assistant

How one admin user goes from signing in to reading a cited, inspectable answer. The two
grey "wait" points are the only places the user pauses: while a document is processed in
the background, and while the answer streams in live.

```mermaid
flowchart TD
    A(["Sign in"]) --> B["Open the app"]
    B --> C{"What do you want<br/>to do?"}

    C -->|Add knowledge| D["Open Documents drawer"]
    D --> E["Upload a file<br/>txt / pdf / docx / csv / json<br/>or paste text"]
    E --> F["Processing in background:<br/>convert, extract, chunk, embed"]:::wait
    F --> G{"Processed OK?"}
    G -->|Failed| H["See a clear error,<br/>fix and retry"]
    H --> E
    G -->|Completed| I["Document is now searchable"]
    I --> C

    C -->|Ask a question| J["Type a question in chat"]
    J --> K["Choose Top K<br/>how many sources to pull in"]
    K --> L{"Top K = 0?"}
    L -->|Yes| M["Answer from the AI's<br/>general knowledge, no citations"]
    L -->|No| N["Answer streams in live"]:::wait
    N --> O["Calculator step shown live<br/>if the question needs math"]
    O --> P["Read the answer<br/>with inline citations like 1, 2"]
    M --> Q{"Want to check<br/>a source?"}
    P --> Q
    Q -->|Click a citation| R["Source drawer opens:<br/>exact passage highlighted,<br/>page / line / row location"]
    R --> S{"Original is a PDF?"}
    S -->|Yes| T["Open the original file<br/>at the cited page"]
    S -->|No| U["Read the extracted passage"]
    T --> V
    U --> V
    Q -->|Keep chatting| V{"Next step?"}
    V -->|Follow-up question| J
    V -->|New topic| W["Start a new conversation"]
    W --> C
    V -->|Done| X(["Sign out /<br/>close the tab"])

    classDef wait fill:#eee,stroke:#999,color:#333,font-style:italic;
```
