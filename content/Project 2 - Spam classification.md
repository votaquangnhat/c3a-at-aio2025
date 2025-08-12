---
title: Project 2 - Spam classification
draft: false
date: 2025-08-02 01:42:18 +0700
---
[## Giới thiệu

Phân loại thư rác (spam) là một vấn đề cổ điển và quan trọng trong xử lý ngôn ngữ tự nhiên (NLP). Mục tiêu là tự động xác định và lọc các tin nhắn không mong muốn, một nhiệm vụ đã trở nên vô cùng cần thiết trong cuộc sống số hàng ngày của chúng ta. Bài blog này trình bày chi tiết hai phương pháp tiếp cận riêng biệt để giải quyết vấn đề này: một phương pháp học máy truyền thống sử dụng Naive Bayes và một phương pháp deep learning tận dụng mô hình tiền huấn luyện và tìm kiếm tương tự FAISS.

### 1. Phương pháp ML truyền thống: Naive Bayes

Phương pháp này sử dụng mô hình Machine Learning truyền thống: Naive Bayes. Các bước ban đầu tập trung vào việc chuyển đổi các tin nhắn văn bản thô thành một định dạng sạch, có cấu trúc phù hợp cho mô hình học máy. Quá trình này bao gồm:

1. **Tiền xử lý**: Mỗi tin nhắn văn bản được thực hiện một loạt các bước làm sạch. Văn bản được chuyển sang chữ thường, sau đó loại bỏ tất cả các dấu câu. Tiếp theo, văn bản được mã hóa thành các từ riêng lẻ. Các từ dừng (stop words) phổ biến trong, sau đó được loại bỏ để tập trung vào các thuật ngữ có ý nghĩa hơn. Cuối cùng, một quá trình rút gọn từ (stemming) được áp dụng để đưa các từ về dạng gốc của chúng.
2. **Tạo vector đặc trưng**: Văn bản đã được tiền xử lý, giờ là một danh sách các từ gốc, được chuyển đổi thành các vector. Hai phương pháp phổ biến đã được sử dụng:
	- **Bag-of-Words (BoW)**: Mô hình này biểu diễn một tài liệu dưới dạng một tập hợp không có thứ tự của các từ, về cơ bản tạo ra một từ vựng từ toàn bộ corpus và đếm tần suất của mỗi từ trong một tin nhắn nhất định. Tuy nhiên, nó bỏ qua tất cả thông tin về thứ tự và ngữ cảnh của từ.
    - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Phương pháp này cân nhắc tần suất một từ xuất hiện trong một tài liệu cụ thể (Term Frequency) và mức độ hiếm của từ đó trong toàn bộ tập hợp tài liệu (Inverse Document Frequency). TF-IDF gán giá trị cao hơn cho những từ quan trọng nhưng ít phổ biến. TF-IDF được kỳ vọng sẽ hoạt động tốt hơn BoW, và điều này đúng khi sử dụng mô hình Gaussian Naive Bayes (sẽ trình bày cụ thể ở phần sau). Xin nhắc lại công thức của TF-IDF:
    
	$$\text{TFIDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{DF(t)}\right)$$
	Trong đó:
	- $TF(t,d)$ (Term Frequency) là số lần thuật ngữ t xuất hiện trong tài liệu d.
	- $N$ là tổng số tài liệu trong corpus D. 
	- $DF(t)$ (Document Frequency) là số tài liệu chứa thuật ngữ t.

3. **Huấn luyện và đánh giá mô hình**: Nhóm đã dùng hai loại mô hình Naive Bayes: **Gaussian Naive Bayes** và **Multinomial Naive Bayes**. Gaussian Naive Bayes giả định rằng các đặc trưng tuân theo phân phối Gaussian (thường dùng cho dữ liệu liên tục), trong khi Multinomial Naive Bayes phù hợp hơn cho các đặc trưng đếm (count-based), như tần suất từ trong văn bản. Các mô hình này được huấn luyện với cả 2 cách tạo vector là BoW và TF-IDF để tìm ra sự kết hợp tốt nhất.

Kết quả đánh giá này rất đáng chú ý. Dưới đây là bảng xếp hạng chi tiết về evaluation của cả bốn sự kết hợp mô hình và cách tạo vector:

| #     | Mô hình & Cách tạo vector | Validation Accuracy | Test Accuracy |
| ----- | ------------------------- | ------------------- | ------------- |
| **1** | **Multinomial NB + BoW**  | **0.9722**          | **0.9695**    |
| 2     | Multinomial NB + TF-IDF   | 0.9632              | 0.9659        |
| 3     | Gaussian NB + TF-IDF      | 0.8780              | 0.8513        |
| 4     | Gaussian NB + BoW         | 0.7354              | 0.7348        |

Như kết quả cho thấy, **Multinomial Naive Bayes** hoạt động vượt trội so với Gaussian Naive Bayes cho bài toán này. Điều này hợp lý vì Multinomial Naive Bayes được thiết kế đặc biệt để xử lý các dữ liệu rời rạc (như số lần xuất hiện của từ), trong khi Gaussian Naive Bayes không phù hợp. Điều thú vị là sự kết hợp giữa **Multinomial NB và BoW** đã cho ra kết quả tốt nhất, vượt qua cả TF-IDF. Điều này có thể là do BoW (Bag of Words) đơn giản nhưng lại nắm bắt tốt hơn sự hiện diện và tần suất của các từ khóa quan trọng trong văn bản đối với mô hình Multinomial Naive Bayes. Mô hình này không quan tâm đến mối quan hệ giữa các từ mà chỉ cần thông tin về tần suất xuất hiện - điều mà BoW cung cấp rõ ràng hơn so với TF-IDF, vốn làm suy giảm trọng số của các từ phổ biến.

Một nguyên nhân khác có thể là do **tập dữ liệu** huấn luyện có đặc điểm mà các từ mang tính phân biệt cao lại cũng thường xuất hiện nhiều lần. Trong trường hợp đó, TF-IDF có thể vô tình làm giảm trọng số những từ có ý nghĩa phân loại mạnh, khiến hiệu quả suy giảm. Trong khi đó, BoW giữ nguyên thông tin thô, giúp mô hình học được mối liên hệ rõ ràng giữa tần suất và nhãn.

### 2. Phương pháp Deep Learning: Dùng pretrained model cho Sentences Embeddings, kết hợp với FAISS

Phương pháp deep learning tận dụng pre-trained model để tạo ra một biểu diễn văn bản giàu ngữ nghĩa hơn.

1. **Mô hình Embedding**: Thay vì chỉ đếm từ đơn giản, phương pháp này sử dụng mô hình `intfloat/multilingual-e5-base` để tạo ra các biểu diễn vector, hay còn gọi là embedding, cho mỗi tin nhắn văn bản. Các embedding này rất mạnh mẽ vì chúng mã hóa ý nghĩa ngữ nghĩa và ngữ cảnh của toàn bộ câu thành một vector duy nhất. Quá trình này bao gồm việc mã hóa các tin nhắn bằng tiền tố `passage:` và sau đó sử dụng mô hình để tạo ra các embedding cuối cùng. Một hàm gọi là `average_pool` đã được sử dụng để lấy các embedding cuối cùng từ đầu ra của mô hình, sau đó được chuẩn hóa.
2. **Cơ sở dữ liệu vector (FAISS)**: Để cho phép tìm kiếm nhanh chóng và hiệu quả các embedding có thứ nguyên cao này, một chỉ mục FAISS (Facebook AI Similarity Search) đã được tạo. Các embedding từ dữ liệu huấn luyện đã được tải vào chỉ mục này.
	Nhóm đã sử dụng **Inner Product (IP)** làm chỉ số đo độ tương đồng giữa các email khi tìm kiếm bằng FAISS. Mặc dù FAISS không hỗ trợ trực tiếp cosine similarity (đo sự giống nhau dựa trên góc giữa các vector), nhưng khi các vector embedding được **chuẩn hóa về độ dài 1 (L2-normalize)**, việc sử dụng IP sẽ cho kết quả tương đương với cosine similarity. Đây là lựa chọn phù hợp cho bài toán phát hiện spam, vì nó sẽ nhấn mạnh hơn vào **ý nghĩa và nội dung** của email thay vì chỉ dựa vào độ dài hay số lượng từ trong email. Và cũng chính vì thế mà khoảng cách L2 không được sử dụng bởi vì nó bị ảnh hưởng nhiều bởi độ lớn của vector.

	**Lợi thế của việc sử dụng FAISS**
	FAISS là một thư viện được tối ưu hóa cao để tìm kiếm tương tự trong các tập dữ liệu vector khổng lồ. Lợi thế của việc sử dụng FAISS ở đây là tốc độ và khả năng mở rộng. Thay vì huấn luyện một mô hình mới để phân loại các embedding, chúng ta có thể đơn giản lưu trữ tất cả các embedding của dữ liệu huấn luyện trong chỉ mục FAISS. Khi một tin nhắn mới, chưa từng thấy đến, chúng ta tạo embedding của nó và sử dụng FAISS để tìm các embedding tương tự nhất từ tập huấn luyện gần như ngay lập tức. Phương pháp này được gọi là phân loại k-láng giềng gần nhất (k-NN). Nó có chi phí tính toán thấp ở thời điểm phân loại và rất hiệu quả vì nó trực tiếp tận dụng thông tin ngữ nghĩa phong phú được nắm bắt bởi mô hình embedding tiền huấn luyện. Nó cũng có nghĩa là chúng ta không phải huấn luyện lại một mô hình deep learning phức tạp để thích ứng với dữ liệu mới; chúng ta chỉ cần cập nhật chỉ mục FAISS với các embedding mới.

3. **Phân loại thông qua k-NN**: Khi một tin nhắn mới cần được phân loại, embedding của nó được tạo ra đầu tiên. Embedding truy vấn này sau đó được sử dụng để tìm kiếm chỉ mục FAISS để tìm `k` láng giềng gần nhất từ dữ liệu huấn luyện. Nhãn phân loại cuối cùng được xác định bằng cách bỏ phiếu đa số trong số `k` láng giềng này.

Hiệu suất của phương pháp này đã được đánh giá trên tập test bằng cách sử dụng các giá trị `k` khác nhau, từ 1 đến 10. Kết quả được tóm tắt trong bảng dưới đây:

|K-value|Độ chính xác|Số lỗi|
|---|---|---|
|1|0.9857|8|
|2|0.9875|7|
|**3**|**0.9928**|**4**|
|4|0.9892|6|
|5|0.9910|5|
|6|0.9892|6|
|7|0.9892|6|
|8|0.9875|7|
|9|0.9875|7|
|10|0.9892|6|

Độ chính xác cao nhất là 99.28% đạt được với giá trị `k` bằng 3. Kết quả này rất xuất sắc, chỉ với 4 lỗi trên 558 mẫu test.

#### Kết luận

Cả hai phương pháp ML truyền thống và deep learning đều cung cấp các giải pháp khá tốt cho việc phân loại thư rác. Kết quả phân tích chi tiết cho thấy rằng mô hình Multinomial Naive Bayes hoạt động hiệu quả hơn đáng kể khi sử dụng BoW, đạt độ chính xác gần 97%. Điều này cho thấy rằng việc chọn mô hình phù hợp với cách tạo vector (ví dụ: Multinomial cho dữ liệu đếm) là rất quan trọng. Tuy nhiên, phương pháp deep learning, với việc sử dụng các embedding của pre-trained model cùng với FAISS, vẫn mang lại hiệu suất cao nhất. Với độ chính xác 99.28%, nó vượt trội hơn đáng kể so với tất cả các mô hình truyền thống đã thử nghiệm. Hiệu quả của phương pháp deep learning nằm ở khả năng nắm bắt ý nghĩa ngữ nghĩa tinh tế của toàn bộ câu, dẫn đến một phân loại mạnh mẽ và chính xác hơn, đặc biệt khi kết hợp với tốc độ và khả năng mở rộng của FAISS.

Code của project này: [votaquangnhat/spam-classification](https://github.com/votaquangnhat/spam-classification)]()