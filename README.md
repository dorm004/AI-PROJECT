# AI-PROJECT

โค้ดเป็นโปรแกรมที่ใช้ Passive Aggressive Classifier เพื่อจำแนกข่าวเป็นข่าวจริงหรือข่าวปลอม โดยใช้ TfidfVectorizer เพื่อแปลงข้อมูลข้อความเป็นเวกเตอร์ที่สามารถใช้ในการฝึกและทดสอบโมเดล และมีการบันทึกและโหลดโมเดลที่ได้ทำการฝึก.

อธิบายขั้นตอนต่าง ๆ ในโค้ด:

1.อ่านข้อมูล:

df=pd.read_csv('D:\\utcc\\Ai\\project\\news.csv'): โค้ดนี้ใช้อ่านข้อมูลจากไฟล์ CSV ที่ตำแหน่ง 'D:\utcc\Ai\project\news.csv' และนำข้อมูลมาเก็บใน DataFrame ที่ชื่อว่า df.

2.แสดงขนาดของข้อมูลและดูหัวตาราง:

df.shape และ df.head(): ใช้เพื่อแสดงขนาดของ DataFrame และดูหัวตารางข้อมูลเพื่อทำความเข้าใจกับโครงสร้างข้อมูล.

3.เก็บป้ายชื่อและแบ่งชุดข้อมูล:

labels=df.label: เก็บป้ายชื่อของข่าว (ข่าวจริงหรือข่าวปลอม).
train_test_split(df['text'], labels, test_size=0.2, random_state=7): แบ่งชุดข้อมูลเป็นชุดฝึกและชุดทดสอบ.

4.สร้างและใช้ TfidfVectorizer:

TfidfVectorizer(stop_words='english', max_df=0.7): สร้าง TfidfVectorizer โดยกำหนดให้ตัดคำที่ไม่มีความหมาย (stop words) ภาษาอังกฤษและกำหนด max_df เพื่อลดความถี่ของคำที่มีความถี่สูง.
tfidf_vectorizer.fit_transform(x_train): ใช้ TfidfVectorizer ในการเรียนรู้และแปลงข้อมูลฝึก.
tfidf_vectorizer.transform(x_test): ใช้ TfidfVectorizer เพื่อแปลงข้อมูลทดสอบ.

5.สร้างและทดสอบ Passive Aggressive Classifier:

PassiveAggressiveClassifier(max_iter=50): สร้าง Passive Aggressive Classifier โดยกำหนด max_iter เป็น 50.
pac.fit(tfidf_train, y_train): ใช้ชุดฝึกข้อมูลในการฝึก Passive Aggressive Classifier.
pac.predict(tfidf_test): ใช้โมเดลที่ฝึกแล้วทำนายข้อมูลทดสอบ.
accuracy_score(y_test, y_pred): คำนวณความแม่นยำของการทำนาย.

6.บันทึกและโหลดโมเดล:

joblib.dump(pac, 'fake_news_classifier_model.joblib'): บันทึก Passive Aggressive Classifier ที่ฝึกไว้ในไฟล์ 'fake_news_classifier_model.joblib'.
joblib.load('fake_news_classifier_model.joblib'): โหลดโมเดลที่บันทึกไว้.

7.ทำนายข้อมูลใหม่:

new_data = [...]: สร้างข้อมูลข่าวใหม่ที่ต้องการทำนาย.
tfidf_vectorizer.transform(new_data): ใช้ TfidfVectorizer เพื่อแปลงข้อมูลใหม่.
pac_loaded.predict(tfidf_new_data): ใช้โมเดลที่โหลดมาทำนายข้อมูลใหม่.
โค้ดนี้ทำการฝึกและทดสอบโมเดลที่ใช้ Passive Aggressive Classifier และ TfidfVectorizer และสามารถทำนายข้อมูลใหม่ได้หลังจากที่บันทึกและโหลดโมเดล.
