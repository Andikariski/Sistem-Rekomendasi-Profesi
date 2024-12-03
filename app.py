from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Path file dataset profesi IT
DATASET_PATH = 'dataset/Dataset_Profesi_ITnew.xlsx'

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        # Membaca dataset profesi IT
        profesi_it = pd.read_excel(DATASET_PATH)
    except FileNotFoundError:
        return render_template('index.html', error="Dataset tidak ditemukan.", skills=[])

    # Validasi dataset
    if "Nama Profesi" not in profesi_it.columns:
        return render_template('index.html', error="Dataset tidak valid. Harus ada kolom 'Nama Profesi'.", skills=[])

    # Ambil semua keterampilan unik dari dataset tanpa menyertakan Nama Profesi
    skill_columns = profesi_it.columns[2:]  # Ambil semua kolom setelah Nama Profesi
    skills = pd.unique(profesi_it[skill_columns].values.ravel())
    skills = [skill for skill in skills if pd.notna(skill)]  # Hapus nilai kosong (NaN)

    if request.method == 'POST':
        # Ambil keterampilan yang dipilih pengguna
        selected_skills = request.form.getlist('skills')

        if not selected_skills:
            return render_template('index.html', error="Pilih minimal satu keterampilan.", skills=skills)

        # Gabungkan keterampilan yang dipilih pengguna
        user_skills = " ".join(selected_skills)

        # Gabungkan keterampilan dari kolom Skill ke satu kolom string(Keterampilan)
        profesi_it["Combined Skills"] = profesi_it[skill_columns].fillna('').apply(
            lambda row: ' '.join(row.astype(str)).strip(), axis=1
        )

        # Ambil keterampilan profesi (tanpa Nama Profesi)
        job_skills = profesi_it["Combined Skills"].tolist()

        # Gabungkan keterampilan pengguna dan profesi IT untuk vectorization
        all_skills = [user_skills] + job_skills

        # Vectorization
        vectorizer = CountVectorizer().fit_transform(all_skills)
        vectors = vectorizer.toarray()

        # Hitung cosine similarity
        cosine_sim = cosine_similarity(vectors[0:1], vectors[1:])

        # Susun rekomendasi (3 teratas)
        recommendations = sorted(
            [(profesi_it.iloc[i]["Nama Profesi"], cosine_sim[0][i]) for i in range(len(profesi_it))],
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Ambil hanya 3 rekomendasi teratas

        return render_template('index.html', recommendations=recommendations, skills=skills)

    return render_template('index.html', skills=skills)


if __name__ == '__main__':
    app.run(debug=True)
