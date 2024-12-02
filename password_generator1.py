import random
import string
import unicodedata

def generate_password(length=12, include_repeats=False, include_language=False, languages=["en"]):
    """Generates a random password of the specified length.

    Args:
        length (int, optional): The length of the password. Defaults to 12.
        include_repeats (bool, optional): Whether to allow repeated characters. Defaults to False.
        include_language (bool, optional): Whether to include characters from specified languages. Defaults to False.
        languages (list, optional): A list of language codes (e.g., "en", "fr", "es") to include characters from. Defaults to ["en"].

    Returns:
        str: The generated password.
    """

    # Base set of characters: letters (uppercase and lowercase), digits, punctuation
    characters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation

    # Add characters for specified languages
    if include_language:
        for language in languages:
            if language == "en":
                continue  # English is already included by default
            elif language == "kn":  # Kannada
                kannada_characters = "\u0C80\u0C81\u0C82\u0C83\u0C85\u0C86\u0C87\u0C88\u0C89\u0C8A\u0C8B\u0C8C\u0C8D\u0C8E\u0C8F\u0C90\u0C91\u0C92\u0C93\u0C94\u0C95\u0C96\u0C97\u0C98\u0C99\u0C9A\u0C9B\u0C9C\u0C9D\u0C9E\u0C9F\u0CA0\u0CA1\u0CA2\u0CA3\u0CA4\u0CA5\u0CA6\u0CA7\u0CA8\u0CA9\u0CAA\u0CAB\u0CAC\u0CAD\u0CAE\u0CAF\u0CB0\u0CB1\u0CB2\u0CB3\u0CB5\u0CB6\u0CB7\u0CB8\u0CB9\u0CBA\u0CBB\u0CBC\u0CBD\u0CBE\u0CBF\u0CC0\u0CC1\u0CC2\u0CC3\u0CC4\u0CC5\u0CC6\u0CC7\u0CC8\u0CC9\u0CCA\u0CCB\u0CCC\u0CCD\u0CCE\u0CCF\u0CD0\u0CD1\u0CD2\u0CD3\u0CD5\u0CD6\u0CD7\u0CD8\u0CD9\u0CDA\u0CDB\u0CDC\u0CDD\u0CDE\u0CDF\u0CE0\u0CE1\u0CE2\u0CE3\u0CE4\u0CE5\u0CE6\u0CE7\u0CE8\u0CE9\u0CEA\u0CEB\u0CEC\u0CED\u0CEE\u0CEF\u0CF0\u0CF1\u0CF2\u0CF3\u0CF5\u0CF6\u0CF7\u0CF8\u0CF9\u0CFA\u0CFB\u0CFC\u0CFD\u0CFE\u0CFF"
                characters += kannada_characters
            elif language == "te":  # Telugu
                telugu_characters = "\u0C00\u0C01\u0C02\u0C03\u0C05\u0C06\u0C07\u0C08\u0C09\u0C0A\u0C0B\u0C0C\u0C0D\u0C0E\u0C0F\u0C10\u0C11\u0C12\u0C13\u0C14\u0C15\u0C16\u0C17\u0C18\u0C19\u0C1A\u0C1B\u0C1C\u0C1D\u0C1E\u0C1F\u0C20\u0C21\u0C22\u0C23\u0C24\u0C25\u0C26\u0C27\u0C28\u0C29\u0C2A\u0C2B\u0C2C\u0C2D\u0C2E\u0C2F\u0C30\u0C31\u0C32\u0C33\u0C35\u0C36\u0C37\u0C38\u0C39\u0C3A\u0C3B\u0C3C\u0C3D\u0C3E\u0C3F\u0C40\u0C41\u0C42\u0C43\u0C44\u0C45\u0C46\u0C47\u0C48\u0C49\u0C4A\u0C4B\u0C4C\u0C4D\u0C4E\u0C4F\u0C50\u0C51\u0C52\u0C53\u0C54\u0C55\u0C56\u0C57\u0C58\u0C59\u0C5A\u0C5B\u0C5C\u0C5D\u0C5E\u0C5F\u0C60\u0C61\u0C62\u0C63\u0C64\u0C65\u0C66\u0C67\u0C68\u0C69\u0C6A\u0C6B\u0C6C\u0C6D\u0C6E\u0C6F\u0C70\u0C71\u0C72\u0C73\u0C75\u0C76\u0C77\u0C78\u0C79\u0C7A\u0C7B\u0C7C\u0C7D\u0C7E"
                characters += telugu_characters

    # Generate password with or without repeats
    if include_repeats:
        password = "".join(random.choices(characters, k=length))
    else:
        password = "".join(random.sample(characters, length))

    return password

if __name__ == "__main__":
    try:
        passlen = int(input("Enter the length of password: "))
        if passlen <= 0:
            raise ValueError("Password length must be positive.")
    except ValueError:
        print("Invalid password length. Using default length of 12.")
        passlen = 12
        

    include_language = input("Include characters from other languages? (y/n): ").lower() == "y"
    if include_language:
        languages = input("Enter language codes (separated by commas): ").split(",")
    else:
        languages = ["en"]  # Default to English

    password = generate_password(passlen, include_repeats=False, include_language=include_language, languages=languages)
    print("Generated password:", password)
