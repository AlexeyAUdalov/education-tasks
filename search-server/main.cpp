#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace std;

/* Подставьте вашу реализацию класса SearchServer сюда */
const int MAX_RESULT_DOCUMENT_COUNT = 5;

vector<string> SplitIntoWords(const string& text) {
    vector<string> words;
    string word;
    for (const char c : text) {
        if (c == ' ') {
            if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        }
        else {
            word += c;
        }
    }
    if (!word.empty()) {
        words.push_back(word);
    }

    return words;
}

struct Document {
    int id;
    double relevance;
    int rating;
};

enum class DocumentStatus {
    ACTUAL,
    IRRELEVANT,
    BANNED,
    REMOVED,
};

class SearchServer {
public:
    void SetStopWords(const string& text) {
        for (const string& word : SplitIntoWords(text)) {
            stop_words_.insert(word);
        }
    }

    void AddDocument(int document_id, const string& document, DocumentStatus status,
        const vector<int>& ratings) {
        const vector<string> words = SplitIntoWordsNoStop(document);
        const double inv_word_count = 1.0 / words.size();

        for (const string& word : words) {
            word_to_document_freqs_[word][document_id] += inv_word_count;
        }

        documents_.emplace(document_id, DocumentData{ ComputeAverageRating(ratings), status });
    }

    template <typename KeyMapper>
    vector<Document> FindTopDocuments(const string& raw_query, KeyMapper key_maper) const {
        const Query query = ParseQuery(raw_query);

        auto matched_documents = FindAllDocuments(query, key_maper);

        sort(matched_documents.begin(), matched_documents.end(),
            [](const Document& lhs, const Document& rhs) {
                if (abs(lhs.relevance - rhs.relevance) < 1e-6) {
                    return lhs.rating > rhs.rating;
                }
                else {
                    return lhs.relevance > rhs.relevance;
                }
            });

        if (matched_documents.size() > MAX_RESULT_DOCUMENT_COUNT) {
            matched_documents.resize(MAX_RESULT_DOCUMENT_COUNT);
        }

        return matched_documents;
    }

    vector<Document> FindTopDocuments(const string& raw_query,
        DocumentStatus output_status = DocumentStatus::ACTUAL) const {
        return FindTopDocuments(raw_query,
            [output_status](int document_id, DocumentStatus status, int rating)
            { return status == output_status; });
    }

    int GetDocumentCount() const {
        return static_cast<int>(documents_.size());
    }

    tuple<vector<string>, DocumentStatus> MatchDocument(const string& raw_query,
        int document_id) const {
        const Query query = ParseQuery(raw_query);
        vector<string> matched_words;
        for (const string& word : query.plus_words) {
            if (word_to_document_freqs_.count(word) == 0) {
                continue;
            }
            if (word_to_document_freqs_.at(word).count(document_id)) {
                matched_words.push_back(word);
            }
        }
        for (const string& word : query.minus_words) {
            if (word_to_document_freqs_.count(word) == 0) {
                continue;
            }
            if (word_to_document_freqs_.at(word).count(document_id)) {
                matched_words.clear();
                break;
            }
        }
        return { matched_words, documents_.at(document_id).status };
    }

private:
    struct DocumentData {
        int rating;
        DocumentStatus status;
    };

    set<string> stop_words_;
    map<string, map<int, double>> word_to_document_freqs_;
    map<int, DocumentData> documents_;

    bool IsStopWord(const string& word) const {
        return stop_words_.count(word) > 0;
    }

    vector<string> SplitIntoWordsNoStop(const string& text) const {
        vector<string> words;
        for (const string& word : SplitIntoWords(text)) {
            if (!IsStopWord(word)) {
                words.push_back(word);
            }
        }
        return words;
    }

    static int ComputeAverageRating(const vector<int>& ratings) {
        if (ratings.empty()) {
            return 0;
        }
        int rating_sum = 0;
        for (const int rating : ratings) {
            rating_sum += rating;
        }
        return rating_sum / static_cast<int>(ratings.size());
    }

    struct QueryWord {
        string data;
        bool is_minus;
        bool is_stop;
    };

    QueryWord ParseQueryWord(string text) const {
        bool is_minus = false;
        // Word shouldn't be empty
        if (text[0] == '-') {
            is_minus = true;
            text = text.substr(1);
        }
        return { text, is_minus, IsStopWord(text) };
    }

    struct Query {
        set<string> plus_words;
        set<string> minus_words;
    };

    Query ParseQuery(const string& text) const {
        Query query;
        for (const string& word : SplitIntoWords(text)) {
            const QueryWord query_word = ParseQueryWord(word);
            if (!query_word.is_stop) {
                if (query_word.is_minus) {
                    query.minus_words.insert(query_word.data);
                }
                else {
                    query.plus_words.insert(query_word.data);
                }
            }
        }
        return query;
    }

    // Existence required
    double ComputeWordInverseDocumentFreq(const string& word) const {
        return log(GetDocumentCount() * 1.0 / word_to_document_freqs_.at(word).size());
    }

    template <typename KeyMapper>
    vector<Document> FindAllDocuments(const Query& query, KeyMapper keymapper) const {
        map<int, double> document_to_relevance;

        for (const string& word : query.plus_words) {
            if (word_to_document_freqs_.count(word) == 0) {
                continue;
            }
            const double inverse_document_freq = ComputeWordInverseDocumentFreq(word);
            for (const auto [document_id, term_freq] : word_to_document_freqs_.at(word)) {
                if (keymapper(document_id, documents_.at(document_id).status,
                    documents_.at(document_id).rating)) {
                    document_to_relevance[document_id] += term_freq * inverse_document_freq;
                }
            }
        }

        for (const string& word : query.minus_words) {
            if (word_to_document_freqs_.count(word) == 0) {
                continue;
            }
            for (const auto [document_id, _] : word_to_document_freqs_.at(word)) {
                document_to_relevance.erase(document_id);
            }
        }

        vector<Document> matched_documents;
        for (const auto [document_id, relevance] : document_to_relevance) {
            matched_documents.push_back(
                { document_id, relevance, documents_.at(document_id).rating });
        }
        return matched_documents;
    }
};



/*
   Подставьте сюда вашу реализацию макросов
   ASSERT, ASSERT_EQUAL, ASSERT_EQUAL_HINT, ASSERT_HINT и RUN_TEST
*/
#define ASSERT_EQUAL(a, b) AssertEqualImpl((a), (b), #a, #b, __FILE__, __FUNCTION__, __LINE__, ""s)
#define ASSERT_EQUAL_HINT(a, b, hint) AssertEqualImpl((a), (b), #a, #b, __FILE__, __FUNCTION__, __LINE__, (hint))
#define ASSERT(a) AssertImpl((a), #a, __FILE__, __FUNCTION__, __LINE__, ""s)
#define ASSERT_HINT(a, hint) AssertImpl((a), #a, __FILE__, __FUNCTION__, __LINE__, (hint))
#define RUN_TEST(func) RunTestImpl(func, #func)


// -------- Начало модульных тестов поисковой системы ----------

template <typename T, typename U>
void AssertEqualImpl(const T& t, const U& u, const string& t_str, const string& u_str, const string& file,
    const string& func, unsigned line, const string& hint) {
    if (t != u) {
        cerr << boolalpha;
        cerr << file << "("s << line << "): "s << func << ": "s;
        cerr << "ASSERT_EQUAL("s << t_str << ", "s << u_str << ") failed: "s;
        cerr << t << " != "s << u << "."s;
        if (!hint.empty()) {
            cerr << " Hint: "s << hint;
        }
        cerr << endl;
        abort();
    }
}

void AssertImpl(bool value, const string& expr_str, const string& file, const string& func, unsigned line,
    const string& hint) {
    if (!value) {
        cerr << file << "("s << line << "): "s << func << ": "s;
        cerr << "Assert("s << expr_str << ") failed."s;
        if (!hint.empty()) {
            cerr << " Hint: "s << hint;
        }
        cerr << endl;
        abort();
    }
}

// Тест проверяет, что поисковая система исключает стоп-слова при добавлении документов
void TestExcludeStopWordsFromAddedDocumentContent() {
    const int doc_id = 42;
    const string content = "cat in the city"s;
    const vector<int> ratings = { 1, 2, 3 };
    {
        SearchServer server;
        server.AddDocument(doc_id, content, DocumentStatus::ACTUAL, ratings);
        const auto found_docs = server.FindTopDocuments("in"s);
        ASSERT_EQUAL(found_docs.size(), 1u);
        const Document& doc0 = found_docs[0];
        ASSERT_EQUAL(doc0.id, doc_id);
    }

    {
        SearchServer server;
        server.SetStopWords("in the"s);
        server.AddDocument(doc_id, content, DocumentStatus::ACTUAL, ratings);
        ASSERT_HINT(server.FindTopDocuments("in"s).empty(),
            "Stop words must be excluded from documents"s);
    }
}

/*
Разместите код остальных тестов здесь
*/

template <typename T>
ostream& operator<<(ostream& output, const vector<T>& items) {
    output << "["s;
    bool first_item = true;
    for (const T& item : items) {
        if (!first_item) {
            output << ", "s;
        }
        output << item;
        first_item = false;
    }
    output << "]"s;
    return output;
}

template <typename T>
ostream& operator<<(ostream& output, const set<T>& items) {
    output << "{"s;
    bool first_item = true;
    for (const T& item : items) {
        if (!first_item) {
            output << ", "s;
        }
        output << item;
        first_item = false;
    }
    output << "}"s;
    return output;
}

template <typename K, typename V>
ostream& operator<<(ostream& output, const map<K, V>& items) {
    output << "{"s;
    bool first_item = true;
    for (const auto& [key, value] : items) {
        if (!first_item) {
            output << ", "s;
        }
        output << key << ": "s << value;
        first_item = false;
    }
    output << "}"s;
    return output;
}

// Добавление документов. 
// Добавленный документ должен находиться по поисковому запросу, который содержит слова из документа.
void TestAddDocumentAndFindAddedDocument() {
    const int doc_id = 3;
    const string content = "the dog is a domesticated descendant of the wolf"s;
    const vector<string> content_words = { "the"s, "dog"s, "is"s, "a"s, "domesticated"s, "descendant"s, "of"s, "the", "wolf"s };
    const vector<int> ratings = { 0, 5, 2 };
    {
        SearchServer server;
        DocumentStatus status = DocumentStatus::ACTUAL;
        server.AddDocument(doc_id, content, DocumentStatus::ACTUAL, ratings);
        for (const auto& word : content_words) {
            const auto found_docs = server.FindTopDocuments(word);
            string hint = word + " was not found in document: "s + content;
            ASSERT_EQUAL_HINT(found_docs.size(), 1u, hint);
            
            const Document& doc0 = found_docs[0];
            hint = word + " was found in document_id="s + to_string(doc0.id)
                + " instead of document_id=" + to_string(doc_id);
            ASSERT_EQUAL_HINT(doc0.id, doc_id, hint);
        }
    }

    const int doc_id_2 = 1;
    const string content_2 = "cat playing with cat"s;
    const vector<string> content_words_2 = { "cat"s, "playing"s, "with"s };
    const vector<int> ratings_2 = { 1 };

    {
        SearchServer server;
        DocumentStatus status = DocumentStatus::ACTUAL;
        server.AddDocument(doc_id, content, DocumentStatus::ACTUAL, ratings);
        server.AddDocument(doc_id_2, content_2, DocumentStatus::ACTUAL, ratings_2);
        for (const auto& word : content_words) {
            const auto found_docs = server.FindTopDocuments(word);
            string hint = word + " was not found in document: "s + content;
            ASSERT_EQUAL_HINT(found_docs.size(), 1u, hint);
            
            const Document& doc0 = found_docs[0];
            hint = word + " was found in document_id="s + to_string(doc0.id)
                + " instead of document_id=" + to_string(doc_id);
            ASSERT_EQUAL_HINT(doc0.id, doc_id, hint);
        }
        for (const auto& word : content_words_2) {
            const auto found_docs = server.FindTopDocuments(word);
            string hint = word + " was not found in document: "s + content;
            ASSERT_EQUAL_HINT(found_docs.size(), 1u, hint);
            
            const Document& doc0 = found_docs[0];
            hint = word + " was found in document_id="s + to_string(doc0.id) 
                   + " instead of document_id=" + to_string(doc_id_2);
            ASSERT_EQUAL_HINT(doc0.id, doc_id_2, hint);
        }
    }
}

//Поддержка минус - слов.
//Документы, содержащие минус - слова из поискового запроса, не должны включаться в результаты поиска.
void TestFindDocumentsWithoutMinusWords() {
    const int doc_id_38 = 38;
    const string content_38 = "cat in the city"s;
    const vector<int> ratings_38 = { 5, 1 };

    const int doc_id_3 = 3;
    const string content_3 = "the dog is a domesticated descendant of the wolf"s;    
    const vector<int> ratings_3 = { 0, 5, 2 };

    const int doc_id_1 = 1;
    const string content_1 = "cat playing with cat"s;
    const vector<int> ratings_1 = { 1 };


    {
        SearchServer server;
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        server.AddDocument(doc_id_3, content_3, DocumentStatus::ACTUAL, ratings_3);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        const auto found_docs = server.FindTopDocuments("dog cat"s);
        string hint = "must be found 3 documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), 3u, hint);
    }
    
    {
        SearchServer server;
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        server.AddDocument(doc_id_3, content_3, DocumentStatus::ACTUAL, ratings_3);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        const auto found_docs = server.FindTopDocuments("dog -cat"s);
        string hint = "must be found 1 document insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), 1u, hint);
        const Document& doc0 = found_docs[0];
        hint = "was found document_id="s + to_string(doc0.id)
            + " instead of document_id=" + to_string(doc_id_3);
        ASSERT_EQUAL_HINT(doc0.id, doc_id_3, hint);
    }
}

//Соответствие документов поисковому запросу. 
// При этом должны быть возвращены все слова из поискового запроса, присутствующие в документе. 
// Если есть соответствие хотя бы по одному минус-слову, должен возвращаться пустой список слов.
void TestFindDocuments() {
    const int doc_id_38 = 38;
    const string content_38 = "cat in the city"s;
    const vector<int> ratings_38 = { 5, 1 };

    const int doc_id_3 = 3;
    const string content_3 = "the dog is a domesticated descendant of the wolf"s;
    const vector<int> ratings_3 = { 0, 5, 2 };

    const int doc_id_1 = 1;
    const string content_1 = "cat playing with cat"s;
    const vector<int> ratings_1 = { 1 };


    {
        SearchServer server;
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        server.AddDocument(doc_id_3, content_3, DocumentStatus::ACTUAL, ratings_3);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        const auto found_docs = server.FindTopDocuments("city wolf cat of is"s);
        const unsigned documents_count = 3u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), 3u, hint);
    }

    {
        SearchServer server;
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        server.AddDocument(doc_id_3, content_3, DocumentStatus::ACTUAL, ratings_3);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        const auto found_docs = server.FindTopDocuments("-cat"s);
        string hint = "must be found 0 document insted of "s + to_string(found_docs.size());
        ASSERT_HINT(static_cast<int>(found_docs.size()) == 0, hint);
    }

    {
        SearchServer server;
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        server.AddDocument(doc_id_3, content_3, DocumentStatus::ACTUAL, ratings_3);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        const auto found_docs = server.FindTopDocuments("city wolf -cat of -is"s);
        string hint = "must be found 0 document insted of "s + to_string(found_docs.size());
        ASSERT_HINT(static_cast<int>(found_docs.size()) == 0, hint);
    }
}

//Сортировка найденных документов по релевантности.
//Возвращаемые при поиске документов результаты должны быть отсортированы в порядке убывания релевантности.
void TestSortingByRelevance() {
    const string stop_words = "is are was a an in the with near at"s;

    const int doc_id_0 = 0;
    const string content_0 = "a colorful parrot with green wings and red tail is lost"s;
    const vector<int> ratings_0 = { 2, -5, -4, 6, 3 };

    const int doc_id_5 = 5;
    const string content_5 = "a white cat with long furry tail is found near the red square"s;
    const vector<int> ratings_5 = { -3, 3, 2, 6 };

    const int doc_id_1 = 1;
    const string content_1 = "a grey hound with black ears is found at the railway station"s;
    const vector<int> ratings_1 = { 7, 9 };

    const int doc_id_38 = 38;
    const string content_38 = "cat in the city"s;
    const vector<int> ratings_38 = { 5, 1 };

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, DocumentStatus::ACTUAL, ratings_0);
        server.AddDocument(doc_id_5, content_5, DocumentStatus::ACTUAL, ratings_5);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        const auto found_docs = server.FindTopDocuments("white cat long tail grey"s);
        const unsigned documents_count = 4u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "was found document_id="s + to_string(doc0.id)
            + " instead of document_id=" + to_string(doc_id_5);
        ASSERT_EQUAL_HINT(doc0.id, doc_id_5, hint);

        const Document& doc1 = found_docs[1];
        hint = "was found document_id="s + to_string(doc1.id)
            + " instead of document_id=" + to_string(doc_id_38);
        ASSERT_EQUAL_HINT(doc1.id, doc_id_38, hint);

        const Document& doc2 = found_docs[2];
        hint = "was found document_id="s + to_string(doc2.id)
            + " instead of document_id=" + to_string(doc_id_1);
        ASSERT_EQUAL_HINT(doc2.id, doc_id_1, hint);

        const Document& doc3 = found_docs[3];
        hint = "was found document_id="s + to_string(doc3.id)
            + " instead of document_id=" + to_string(doc_id_0);
        ASSERT_EQUAL_HINT(doc3.id, doc_id_0, hint);
    }

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, DocumentStatus::ACTUAL, ratings_0);
        server.AddDocument(doc_id_5, content_5, DocumentStatus::ACTUAL, ratings_5);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        const auto found_docs = server.FindTopDocuments("green red grey"s);
        const unsigned documents_count = 3u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "was found document_id="s + to_string(doc0.id)
            + " instead of document_id=" + to_string(doc_id_0);
        ASSERT_EQUAL_HINT(doc0.id, doc_id_0, hint);

        const Document& doc1 = found_docs[1];
        hint = "was found document_id="s + to_string(doc1.id)
            + " instead of document_id=" + to_string(doc_id_1);
        ASSERT_EQUAL_HINT(doc1.id, doc_id_1, hint);

        const Document& doc2 = found_docs[2];
        hint = "was found document_id="s + to_string(doc2.id)
            + " instead of document_id=" + to_string(doc_id_5);
        ASSERT_EQUAL_HINT(doc2.id, doc_id_5, hint);
    }    
}

//Вычисление рейтинга документов.
//Рейтинг добавленного документа равен среднему арифметическому оценок документа.
void TestRatingCalculation() {
    const int doc_id_0 = 0;
    const string content_0 = "a colorful parrot with green wings and red tail is lost"s;
    const vector<int> ratings_0 = { 2, -5, -4, 6, 3 };
    const int calculate_rating_0 = 0;

    const int doc_id_5 = 5;
    const string content_5 = "a white cat with long furry tail is found near the red square"s;
    const vector<int> ratings_5 = { -3, 3, 2, 6 };
    const int calculate_rating_5 = 2;

    const int doc_id_1 = 1;
    const string content_1 = "a grey hound with black ears is found at the railway station"s;
    const vector<int> ratings_1 = { 7, -9, -4 };
    const int calculate_rating_1 = -2;

    {
        SearchServer server;
        server.AddDocument(doc_id_0, content_0, DocumentStatus::ACTUAL, ratings_0);
        const auto found_docs = server.FindTopDocuments("colorful"s);
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "document rating is "s + to_string(doc0.rating)
            + " instead of rating " + to_string(calculate_rating_0);
        ASSERT_EQUAL_HINT(doc0.rating, calculate_rating_0, hint);
    }

    {
        SearchServer server;
        server.AddDocument(doc_id_5, content_5, DocumentStatus::ACTUAL, ratings_5);
        const auto found_docs = server.FindTopDocuments("long"s);
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "document rating is "s + to_string(doc0.rating)
            + " instead of rating " + to_string(calculate_rating_5);
        ASSERT_EQUAL_HINT(doc0.rating, calculate_rating_5, hint);
    }

    {
        SearchServer server;
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        const auto found_docs = server.FindTopDocuments("found"s);
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "document rating is "s + to_string(doc0.rating)
            + " instead of rating " + to_string(calculate_rating_1);
        ASSERT_EQUAL_HINT(doc0.rating, calculate_rating_1, hint);
    }
}

//Фильтрация результатов поиска с использованием предиката, задаваемого пользователем.
void TestPredicate() {
    const string stop_words = "is are was a an in the with near at"s;

    const int doc_id_0 = 0;
    const string content_0 = "a colorful parrot with green wings and red tail is lost"s;
    const vector<int> ratings_0 = { 2, -5, -4, 6, 3 };
    const int calculate_rating_0 = 0;

    const int doc_id_5 = 5;
    const string content_5 = "a white cat with long furry tail is found near the red square"s;
    const vector<int> ratings_5 = { -3, 3, 2, -6 };
    const int calculate_rating_5 = -1;

    const int doc_id_1 = 1;
    const string content_1 = "a grey hound with black ears is found at the railway station"s;
    const vector<int> ratings_1 = { 7, 9 };
    const int calculate_rating_1 = 8;

    const int doc_id_38 = 38;
    const string content_38 = "cat in the city"s;
    const vector<int> ratings_38 = { 5, 1 };
    const int calculate_rating_38 = 3;

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, DocumentStatus::ACTUAL, ratings_0);
        server.AddDocument(doc_id_5, content_5, DocumentStatus::ACTUAL, ratings_5);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        const auto found_docs = server.FindTopDocuments("white cat long tail grey"s,
            [](int document_id, DocumentStatus status, int rating) { return rating > 0; });
        const unsigned documents_count = 2u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "document rating is "s + to_string(doc0.rating)
            + " instead of rating " + to_string(calculate_rating_38);
        ASSERT_EQUAL_HINT(doc0.rating, calculate_rating_38, hint);

        const Document& doc1 = found_docs[1];
        hint = "document rating is "s + to_string(doc1.rating)
            + " instead of rating " + to_string(calculate_rating_1);
        ASSERT_EQUAL_HINT(doc1.rating, calculate_rating_1, hint);
    }

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, DocumentStatus::ACTUAL, ratings_0);
        server.AddDocument(doc_id_5, content_5, DocumentStatus::ACTUAL, ratings_5);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        const auto found_docs = server.FindTopDocuments("white cat long tail grey"s,
            [](int document_id, DocumentStatus status, int rating) { return document_id == 1; });
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "was found document_id="s + to_string(doc0.id)
            + " instead of document_id=" + to_string(doc_id_1);
        ASSERT_EQUAL_HINT(doc0.id, doc_id_1, hint);
    }

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, DocumentStatus::ACTUAL, ratings_0);
        server.AddDocument(doc_id_5, content_5, DocumentStatus::BANNED, ratings_5);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        const auto found_docs = server.FindTopDocuments("white cat long tail grey"s,
            [](int document_id, DocumentStatus status, int rating) { return status == DocumentStatus::BANNED; });
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "was found document_id="s + to_string(doc0.id)
            + " instead of document_id=" + to_string(doc_id_5);
        ASSERT_EQUAL_HINT(doc0.id, doc_id_5, hint);
    }
}

//Поиск документов, имеющих заданный статус
void TestSearchingByStatus() {
    const string stop_words = "is are was a an in the with near at"s;

    const int doc_id_0 = 0;
    const string content_0 = "a colorful parrot with green wings and red tail is lost"s;
    const vector<int> ratings_0 = { 2, -5, -4, 6, 3 };
    const DocumentStatus status_0 = DocumentStatus::ACTUAL;

    const int doc_id_5 = 5;
    const string content_5 = "a white cat with long furry tail is found near the red square"s;
    const vector<int> ratings_5 = { -3, 3, 2, -6 };
    const DocumentStatus status_5 = DocumentStatus::BANNED;

    const int doc_id_1 = 1;
    const string content_1 = "a grey hound with black ears is found at the railway station"s;
    const vector<int> ratings_1 = { 7, 9 };
    const DocumentStatus status_1 = DocumentStatus::REMOVED;

    const int doc_id_38 = 38;
    const string content_38 = "cat in the city"s;
    const vector<int> ratings_38 = { 5, 1 };
    const DocumentStatus status_38 = DocumentStatus::IRRELEVANT;

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, status_0, ratings_0);
        server.AddDocument(doc_id_5, content_5, status_5, ratings_5);
        server.AddDocument(doc_id_1, content_1, status_1, ratings_1);
        server.AddDocument(doc_id_38, content_38, status_38, ratings_38);
        const auto found_docs = server.FindTopDocuments("white cat long tail grey"s, status_0);
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "was found document_id="s + to_string(doc0.id)
            + " instead of document_id=" + to_string(doc_id_0);
        ASSERT_EQUAL_HINT(doc0.id, doc_id_0, hint);
    }

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, status_0, ratings_0);
        server.AddDocument(doc_id_5, content_5, status_5, ratings_5);
        server.AddDocument(doc_id_1, content_1, status_1, ratings_1);
        server.AddDocument(doc_id_38, content_38, status_38, ratings_38);
        const auto found_docs = server.FindTopDocuments("white cat long tail grey"s, status_5);
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "was found document_id="s + to_string(doc0.id)
            + " instead of document_id=" + to_string(doc_id_5);
        ASSERT_EQUAL_HINT(doc0.id, doc_id_5, hint);
    }

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, status_0, ratings_0);
        server.AddDocument(doc_id_5, content_5, status_5, ratings_5);
        server.AddDocument(doc_id_1, content_1, status_1, ratings_1);
        server.AddDocument(doc_id_38, content_38, status_38, ratings_38);
        const auto found_docs = server.FindTopDocuments("white cat long tail grey"s, status_1);
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "was found document_id="s + to_string(doc0.id)
            + " instead of document_id=" + to_string(doc_id_1);
        ASSERT_EQUAL_HINT(doc0.id, doc_id_1, hint);
    }

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, status_0, ratings_0);
        server.AddDocument(doc_id_5, content_5, status_5, ratings_5);
        server.AddDocument(doc_id_1, content_1, status_1, ratings_1);
        server.AddDocument(doc_id_38, content_38, status_38, ratings_38);
        const auto found_docs = server.FindTopDocuments("white cat long tail grey"s, status_38);
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const Document& doc0 = found_docs[0];
        hint = "was found document_id="s + to_string(doc0.id)
            + " instead of document_id=" + to_string(doc_id_38);
        ASSERT_EQUAL_HINT(doc0.id, doc_id_38, hint);
    }
}

//Корректное вычисление релевантности найденных документов
void TestRelevanceCalculation() {
    const double epsilon_value = 1e-6;

    const string stop_words = "is are was a an in the with near at"s;

    const int doc_id_0 = 0;
    const string content_0 = "a colorful parrot with green wings and red tail is lost cat"s;
    const vector<int> ratings_0 = { 2, -5, -4, 6, 3 };

    const int doc_id_5 = 5;
    const string content_5 = "a white cat with long furry tail is found near the red square"s;
    const vector<int> ratings_5 = { -3, 3, 2, -6 };

    const int doc_id_1 = 1;
    const string content_1 = "a grey hound with black ears is found at the railway station cat"s;
    const vector<int> ratings_1 = { 7, 9 };

    const int doc_id_38 = 38;
    const string content_38 = "cat in the city"s;
    const vector<int> ratings_38 = { 5, 1 };

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, DocumentStatus::ACTUAL, ratings_0);
        server.AddDocument(doc_id_5, content_5, DocumentStatus::ACTUAL, ratings_5);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        const auto found_docs = server.FindTopDocuments("cat"s);
        const unsigned documents_count = 4u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const double relevance = 0.;
        const Document& doc0 = found_docs[0];        
        hint = "document_id=" + to_string(doc0.id) + " rating is "s + to_string(doc0.relevance)
            + " instead of rating " + to_string(relevance);
        ASSERT_HINT(abs(doc0.relevance - relevance) < epsilon_value, hint);

        const Document& doc1 = found_docs[1];
        hint = "document_id=" + to_string(doc1.id) + " rating is "s + to_string(doc1.relevance)
            + " instead of rating " + to_string(relevance);
        ASSERT_HINT(abs(doc1.relevance - relevance) < epsilon_value, hint);

        const Document& doc2 = found_docs[2];
        hint = "document_id=" + to_string(doc1.id) + " rating is "s + to_string(doc2.relevance)
            + " instead of rating " + to_string(relevance);
        ASSERT_HINT(abs(doc2.relevance - relevance) < epsilon_value, hint);

        const Document& doc3 = found_docs[3];
        hint = "document_id=" + to_string(doc1.id) + " rating is "s + to_string(doc3.relevance)
            + " instead of rating " + to_string(relevance);
        ASSERT_HINT(abs(doc3.relevance - relevance) < epsilon_value, hint);
    }

    {
        SearchServer server;
        server.SetStopWords("is are was a an in the with near at"s);
        server.AddDocument(doc_id_0, content_0, DocumentStatus::ACTUAL, ratings_0);
        server.AddDocument(doc_id_5, content_5, DocumentStatus::ACTUAL, ratings_5);
        server.AddDocument(doc_id_1, content_1, DocumentStatus::ACTUAL, ratings_1);
        server.AddDocument(doc_id_38, content_38, DocumentStatus::ACTUAL, ratings_38);
        const auto found_docs = server.FindTopDocuments("green"s);
        const unsigned documents_count = 1u;
        string hint = "must be found " + to_string(documents_count) + " documents insted of "s + to_string(found_docs.size());
        ASSERT_EQUAL_HINT(found_docs.size(), documents_count, hint);

        const double relevance = 0.154033;
        const Document& doc0 = found_docs[0];
        hint = "document_id=" + to_string(doc0.id) + " rating is "s + to_string(doc0.relevance)
            + " instead of rating " + to_string(relevance);
        ASSERT_HINT(abs((doc0.relevance - relevance) < epsilon_value) && doc0.id == doc_id_0, hint);
    }
}






template <typename TestFunc>
void RunTestImpl(const TestFunc& func, const string& test_name) {
    func();
    cerr << test_name << " OK"s << endl;
}


// Функция TestSearchServer является точкой входа для запуска тестов
void TestSearchServer() {
    RUN_TEST(TestExcludeStopWordsFromAddedDocumentContent);
    // Не забудьте вызывать остальные тесты здесь
    RUN_TEST(TestAddDocumentAndFindAddedDocument);
    RUN_TEST(TestFindDocumentsWithoutMinusWords);
    RUN_TEST(TestFindDocuments);
    RUN_TEST(TestSortingByRelevance);
    RUN_TEST(TestRatingCalculation);
    RUN_TEST(TestPredicate);
    RUN_TEST(TestSearchingByStatus);
    RUN_TEST(TestRelevanceCalculation);
}

// --------- Окончание модульных тестов поисковой системы -----------

int main() {
    TestSearchServer();
    // Если вы видите эту строку, значит все тесты прошли успешно
    cout << "Search server testing finished"s << endl;
}
