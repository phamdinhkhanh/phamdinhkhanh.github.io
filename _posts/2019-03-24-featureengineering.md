---
layout: post
author: phamdinhkhanh
title: Kĩ thuật feature engineering
---

1. Giới thiệu về feature engineering
Hiện nay các phương pháp học máy xuất hiện ngày càng nhiều và trở nên mạnh mẽ hơn. 
Các mô hình học máy như mạng neural network, Random Forest, Decision Tree, SVM, kNN,... đều
 là những mô hình có tính tường minh thấp, độ chính xác cao, độ phức tạp và tính linh hoạt cao. 
 Các mô hình học máy đa dạng sẽ làm phong phú thêm sự lựa chọn của các modeler. 
 Tuy nhiên bên cạnh việc áp dụng các phương pháp mạnh, modeler cần phải chuẩn hóa dữ liệu tốt, 
 bởi dữ liệu là nguyên liệu để mô hình dựa trên đó xây dựng một phương pháp học. Nếu mô hình học trên một bộ dữ liệu không tốt, 
 kết quả dự báo sẽ không tốt. Nếu mô hình học trên một bộ dữ liệu trúng tủ, kết quả mô hình sẽ được cải thiện. 
 Chính vì thế vai trò của chuẩn hóa dữ liệu quan trọng đến mức Andrew Nguyen đã từng nói 'xây dựng mô hình machine learning không gì khác
 là thực hiện feature engineering'. Và thực tế cũng cho thấy trong các cuộc thi phân tích dữ liệu, các leader board đều áp dụng tốt các kĩ 
 thuật tạo đặc trưng bên cạnh việc áp dụng những phương pháp mạnh. Những mô hình đơn giản nhưng được xây dựng trên biến chất lượng thường 
 mang lại hiệu quả hơn những mô hình phức tạp như mạng nơ ron hoặc các mô hình kết hợp nhưng được xây dựng trên biến chưa được sử dụng các
 kĩ thuật tạo đặc trưng.

Về kĩ thuật tạo đặc trưng chúng ta có 3 phương pháp chính:
