```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant F as Frontend
    participant A as API
    participant S as Storage Cloud
    participant B as BDD

    U->>F: Sélectionne image
    F->>F: Validation client (taille, type)
    F->>A: POST /images avec FormData
    A->>A: Génère nom unique
    A->>S: Upload vers Cloud Storage
    S-->>A: URL publique
    A->>B: UPDATE recipe SET image_url = ?
    B-->>A: Confirmation
    A-->>F: 200 OK + URL image
    F-->>U: Affiche image preview