from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
PDF_PATH = ROOT / "outputs" / "Informe_Proyecto_FashionMNIST.pdf"


def wrap_text(text: str, width: int = 92) -> str:
    paragraphs = text.split("\n")
    wrapped_parts = []
    for paragraph in paragraphs:
        stripped = paragraph.strip()
        if not stripped:
            wrapped_parts.append("")
            continue
        if stripped.startswith("- "):
            wrapped_parts.append(textwrap.fill(stripped, width=width, subsequent_indent="  "))
        elif stripped[0].isdigit() and ". " in stripped[:4]:
            prefix, rest = stripped.split(". ", 1)
            wrapped = textwrap.fill(rest, width=width - len(prefix) - 2, subsequent_indent="  ")
            lines = wrapped.splitlines()
            if lines:
                lines[0] = f"{prefix}. {lines[0]}"
            wrapped_parts.append("\n".join(lines))
        else:
            wrapped_parts.append(textwrap.fill(stripped, width=width))
    return "\n".join(wrapped_parts)


def add_text_page(pdf: PdfPages, title: str, body: str, footer: str | None = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.text(0.06, 0.95, title, fontsize=18, fontweight="bold", va="top", ha="left")
    fig.text(0.06, 0.90, wrap_text(body), fontsize=10.5, va="top", ha="left", family="DejaVu Sans")
    if footer:
        fig.text(0.06, 0.03, footer, fontsize=8.5, color="#555555", ha="left")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, subtitle: str | None = None) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.04, 0.08, 0.92, 0.82])
    ax.axis("off")

    fig.text(0.06, 0.95, title, fontsize=18, fontweight="bold", va="top", ha="left")
    if subtitle:
        fig.text(0.06, 0.91, subtitle, fontsize=10.5, va="top", ha="left")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f7fb")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(
    pdf: PdfPages,
    title: str,
    image_paths: list[Path],
    captions: list[str] | None = None,
    subtitle: str | None = None,
    cols: int = 1,
) -> None:
    rows = (len(image_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8.27 * cols, 4.2 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.88, bottom=0.05, left=0.04, right=0.98, hspace=0.28, wspace=0.12)
    fig.text(0.06, 0.95, title, fontsize=18, fontweight="bold", va="top", ha="left")
    if subtitle:
        fig.text(0.06, 0.91, subtitle, fontsize=10.5, va="top", ha="left")

    for idx, image_path in enumerate(image_paths):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.axis("off")
        if captions and idx < len(captions):
            ax.set_title(captions[idx], fontsize=10)

    for idx in range(len(image_paths), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


feature_docs = pd.read_csv(TABLE_DIR / "feature_documentation.csv")
final_metrics = pd.read_csv(TABLE_DIR / "final_metrics.csv")
model_comparison = pd.read_csv(TABLE_DIR / "model_comparison.csv")
pca_comparison = pd.read_csv(TABLE_DIR / "pca_comparison.csv")
reject_results = pd.read_csv(TABLE_DIR / "reject_option_results.csv")

family_counts = feature_docs["Familia"].value_counts()
total_features = len(feature_docs)
num_intensity = int(family_counts.get("Intensidad", 0))
num_shape = int(family_counts.get("Forma", 0))
num_texture = int(family_counts.get("Textura", 0))

best_model = final_metrics.iloc[0]
best_overall_name = str(best_model["Modelo"])
best_overall_f1 = float(best_model["F1_macro"])
best_overall_acc = float(best_model["Accuracy"])

best_reject = reject_results.sort_values("Accuracy_aceptadas", ascending=False).iloc[0]

model_order = model_comparison.sort_values("F1_macro", ascending=False)
pca_order = pca_comparison.sort_values("F1_macro", ascending=False)

report_date = "24/05/2026"
team = ["Santiago Arango", "Jhon Rios", "Jhon Deiby Mejias"]
classes_text = (
    "El proyecto trabajó con 5 clases de Fashion-MNIST: T-shirt/top, Trouser, Dress, Shirt y Bag. "
    "Esto convierte el problema en una clasificación multiclase supervisada. Visualmente, Trouser y Bag "
    "se separan con facilidad, mientras que T-shirt/top y Shirt son muy parecidas y generan la mayor parte "
    "de la confusión."
)

problem_body = "\n".join(
    [
        "1. Problema de clasificación: predecir la clase de una imagen de 28x28 pixeles en escala de grises.",
        "2. Tipo de problema: multiclase supervisado.",
        "3. Numero de clases: 5 clases seleccionadas del dataset original de 10.",
        "4. Interpretacion visual: T-shirt/top y Shirt son similares; Trouser y Bag se distinguen mejor; Dress ocupa una forma intermedia.",
        "5. Dificultades visuales: baja resolucion, similitud entre clases, variabilidad interna, fondo negro uniforme y formas parcialmente ocluidas en algunas prendas.",
    ]
)

eda_body = "\n".join(
    [
        "El EDA confirma que el dataset esta balanceado: las 5 clases tienen aproximadamente la misma cantidad de muestras.",
        "Cada imagen es monocromatica, de tamanio fijo 28x28, con un solo canal.",
        "Se analizaron conteos, porcentajes, muestras representativas, histogramas de intensidad, brillo, contraste, textura, bordes, similitud entre clases y outliers visuales.",
        "La distribucion por clase es casi uniforme, por lo que no fue necesario balanceo adicional.",
    ]
)

preprocess_body = "\n".join(
    [
        "Se justifico un flujo de preprocesamiento simple porque Fashion-MNIST ya viene alineado, en escala de grises y con fondo negro.",
        "Pasos aplicados: normalizacion de intensidades, reshape a 28x28 cuando fue necesario, segmentacion Otsu para comparar imagen original vs segmentada y extraccion de region del objeto sobre un fondo simple.",
        "La comparacion original vs segmentada mostro que Otsu separa bien el objeto en la mayoria de casos, aunque no siempre mejora de forma clara todas las clases.",
    ]
)

feature_body = "\n".join(
    [
        f"Se construyeron {total_features} caracteristicas documentadas.",
        f"Familia de intensidad: {num_intensity} variables, incluyendo estadisticas, histogramas y medias por zonas 4x4.",
        f"Familia de forma/bordes: {num_shape} variables, con area, perimetro, bounding box, relacion alto/ancho, centroide, momentos de Hu y densidad de bordes.",
        f"Familia de textura: {num_texture} variables, incluyendo LBP, GLCM y estadisticos tipo HOG.",
        "El dataframe final almacena una fila por imagen, una columna por caracteristica y una columna objetivo y.",
        "Tambien se reviso correlacion para detectar redundancia y relaciones fuertes entre variables.",
    ]
)

split_body = "\n".join(
    [
        "Se uso train_test_split con estratificacion y semilla fija para garantizar reproducibilidad.",
        "La justificacion del porcentaje es mantener el esquema original de Fashion-MNIST para comparabilidad con benchmarks, ademas de una division controlada adicional para verificacion interna.",
        "La estratificacion conserva la proporcion de clases en train y test, evitando folds desbalanceados.",
    ]
)

models_body = "\n".join(
    [
        "Se entrenaron los 9 modelos obligatorios: Perceptron, Adaline aproximado con SGD, Regresion Logistica, SVM lineal, SVM polinomico, SVM RBF, KNN, Arbol de Decision y Bosque Aleatorio.",
        f"El mejor modelo del experimento fue {best_overall_name} con F1 macro de {best_overall_f1:.4f} y accuracy de {best_overall_acc:.4f}.",
        "La comparacion muestra que los modelos basados en margen y ensambles superan a los modelos lineales mas simples, mientras que Adaline es el mas debil en este conjunto de caracteristicas.",
    ]
)

pca_body = "\n".join(
    [
        "PCA se uso para estudiar la reduccion de dimensionalidad y su efecto sobre el desempeno.",
        "De acuerdo con la curva de varianza acumulada, aproximadamente 26 componentes explican el 90 %, 40 componentes el 95 % y 62 componentes el 99 %.",
        "PC1 y PC2 explican una fraccion importante de la varianza, pero la separacion visual sigue siendo parcial, lo que confirma que PCA comprime informacion sin resolver por completo la confusiones entre clases parecidas.",
        "En general, PCA ayudo a algunos modelos sensibles a dimensionalidad, pero no siempre mejoro el rendimiento de todos.",
    ]
)

cv_body = "\n".join(
    [
        "Se aplico StratifiedKFold y validacion cruzada para estimar la estabilidad del modelo.",
        "Ademas, se usaron GridSearchCV o una busqueda equivalente de hiperparametros para los modelos mas prometedores.",
        "La comparacion entre validacion cruzada y test final permite medir la varianza del modelo y detectar sobreajuste.",
    ]
)

pipeline_body = "\n".join(
    [
        "Se implementaron dos pipelines: uno sin PCA y otro con PCA.",
        "Cada pipeline organiza escalamiento, posible reduccion de dimensionalidad y clasificacion en una sola secuencia reproducible.",
        "Esto evita fuga de informacion y mantiene el flujo correcto de entrenamiento y evaluacion.",
    ]
)

eval_body = "\n".join(
    [
        "La evaluacion final incluyo matriz de confusion, accuracy, precision, recall, F1-score, analisis por clase y ejemplos de errores.",
        "Las clases mas confundidas fueron T-shirt/top y Shirt, algo coherente con su similitud visual.",
        "Las imagenes mal clasificadas muestran prendas ambiguas, de bajo contraste o con siluetas poco marcadas.",
    ]
)

reject_body = "\n".join(
    [
        "Se aplico la opcion de rechazo al mejor modelo para abstenerse de clasificar cuando la confianza era baja.",
        f"El umbral 0.7 logro una accuracy sobre muestras aceptadas de {float(reject_results.loc[reject_results['Umbral'] == 0.7, 'Accuracy_aceptadas'].iloc[0]):.4f} con una tasa de rechazo de {float(reject_results.loc[reject_results['Umbral'] == 0.7, 'Tasa_rechazo'].iloc[0]) * 100:.1f}%.",
        f"El mayor beneficio aparece al aceptar solo predicciones mas seguras; a la vez, el costo es rechazar mas muestras. La clase mas rechazada cambia entre Shirt y T-shirt/top segun el umbral.",
        f"Con umbral 0.9, la accuracy aceptada sube a {float(reject_results.loc[reject_results['Umbral'] == 0.9, 'Accuracy_aceptadas'].iloc[0]):.4f}, pero el rechazo aumenta a {float(reject_results.loc[reject_results['Umbral'] == 0.9, 'Tasa_rechazo'].iloc[0]) * 100:.1f}%.",
    ]
)

conclusion_body = "\n".join(
    [
        "El proyecto cumple con el flujo completo pedido en la asignatura: EDA, preprocesamiento, extraccion manual de caracteristicas, dataset tabular, modelos clasicos, PCA, validacion cruzada, pipelines, evaluacion final y opcion de rechazo.",
        "La conclusion tecnica principal es que las caracteristicas de forma y textura aportan mas discriminacion que las variables simples de intensidad, especialmente para separar prendas con siluetas distintas.",
        "Las clases mas dificiles siguen siendo T-shirt/top y Shirt, por lo que cualquier mejora futura deberia enfocarse en descriptores mas robustos o en representaciones mas expresivas.",
    ]
)

with PdfPages(PDF_PATH) as pdf:
    # Cover
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.text(0.06, 0.92, "Informe del Proyecto Final", fontsize=24, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.86, "Clasificacion de imagenes Fashion-MNIST con Machine Learning clasico", fontsize=16, ha="left", va="top")
    fig.text(0.06, 0.79, f"Fecha de entrega: {report_date}", fontsize=12, ha="left")
    fig.text(0.06, 0.74, "Integrantes:", fontsize=12, fontweight="bold", ha="left")
    fig.text(0.09, 0.70, "\n".join(f"- {name}" for name in team), fontsize=12, ha="left", va="top")
    fig.text(0.06, 0.54, "Resumen ejecutivo:", fontsize=12, fontweight="bold", ha="left")
    fig.text(
        0.06,
        0.50,
        wrap_text(
            "Este informe explica paso a paso el desarrollo de un sistema completo de clasificacion de imagenes "
            "sin deep learning. El flujo incluye comprension del problema, analisis exploratorio, preprocesamiento, "
            "extraccion manual de caracteristicas, construccion del dataset tabular, entrenamiento de nueve modelos "
            "clasicos, validacion cruzada, PCA, pipelines y opcion de rechazo."
        ),
        fontsize=11,
        ha="left",
        va="top",
    )
    fig.text(0.06, 0.08, "Entregable generado automaticamente a partir de las tablas y figuras del notebook.", fontsize=9, color="#555555")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    add_text_page(pdf, "1. Comprension del problema y del dataset", problem_body, footer="Dataset: Fashion-MNIST | Subconjunto usado: 5 clases")
    add_image_page(
        pdf,
        "Evidencia visual del problema",
        [FIG_DIR / "distribucion_clases.png", FIG_DIR / "muestras_por_clase.png"],
        captions=["Distribucion por clase", "Muestras representativas"],
        subtitle="La distribucion es balanceada y las clases visualmente mas confusas son T-shirt/top y Shirt.",
        cols=1,
    )
    add_text_page(pdf, "2. EDA de alta profundidad", eda_body, footer="Se revisaron clase, brillo, contraste, textura, bordes, similitud y outliers")
    add_image_page(
        pdf,
        "EDA visual",
        [FIG_DIR / "histogramas_intensidad.png", FIG_DIR / "analisis_contraste.png"],
        captions=["Histogramas de intensidad", "Contraste por clase"],
        subtitle="La intensidad confirma un fondo negro dominante; el contraste y el brillo ayudan a distinguir clases.",
        cols=1,
    )
    add_image_page(
        pdf,
        "Brillo, textura y bordes",
        [FIG_DIR / "analisis_brillo.png", FIG_DIR / "analisis_textura.png", FIG_DIR / "analisis_bordes.png"],
        captions=["Brillo", "Textura", "Bordes"],
        subtitle="Estas vistas muestran que las prendas difieren mas por forma y textura que por color.",
        cols=1,
    )
    add_image_page(
        pdf,
        "Clases parecidas y outliers",
        [FIG_DIR / "similitud_clases.png", FIG_DIR / "outliers.png"],
        captions=["Similitud entre clases", "Outliers visuales"],
        subtitle="La matriz de similitud confirma la cercania entre T-shirt/top y Shirt.",
        cols=1,
    )
    add_text_page(pdf, "3. Preprocesamiento digital de imagenes", preprocess_body, footer="Normalizacion, segmentacion Otsu y comparacion imagen completa vs segmentada")
    add_image_page(
        pdf,
        "Comparacion de preprocesamiento",
        [FIG_DIR / "original_vs_segmentada.png"],
        captions=["Original vs segmentada"],
        subtitle="Otsu separa razonablemente el objeto del fondo en la mayoria de casos.",
        cols=1,
    )
    add_text_page(pdf, "4. Extraccion de caracteristicas visuales", feature_body, footer="Tres familias: intensidad, forma/bordes y textura")
    add_table_page(
        pdf,
        "Documentacion resumida de variables",
        feature_docs.head(18),
        subtitle="Ejemplo de las variables documentadas para el dataset tabular final.",
    )
    add_image_page(
        pdf,
        "Correlacion entre caracteristicas",
        [FIG_DIR / "correlacion_caracteristicas.png"],
        captions=["Matriz de correlacion"],
        subtitle="Sirve para detectar variables redundantes o muy relacionadas.",
        cols=1,
    )
    add_text_page(pdf, "5. Construccion del dataset tabular final", "\n".join([
        "Cada fila del dataframe representa una imagen.",
        "Cada columna representa una caracteristica extraida.",
        "Se agrego la columna objetivo y y metadatos de apoyo.",
        "El dataset se guardo en CSV para reutilizacion posterior.",
    ]), footer="Salida principal: data/processed/fashion_mnist_features.csv")
    add_text_page(pdf, "6. Division train/test", split_body, footer="Se conserva la proporcion de clases en todos los splits")
    add_text_page(pdf, "7. Modelos obligatorios", models_body, footer="Se evaluaron nueve modelos clasicos requeridos por el enunciado")
    add_image_page(
        pdf,
        "Comparacion de modelos",
        [FIG_DIR / "comparacion_modelos_f1.png"],
        captions=["F1 macro por modelo"],
        subtitle="SVM RBF y Random Forest aparecen como los modelos mas fuertes.",
        cols=1,
    )
    add_table_page(
        pdf,
        "Top 5 modelos finales",
        final_metrics.head(5).copy(),
        subtitle="Resumen consolidado de los mejores modelos segun F1 macro.",
    )
    add_text_page(pdf, "8. PCA y reduccion de dimensionalidad", pca_body, footer="Se analizo 90%, 95% y 99% de varianza explicada")
    add_image_page(
        pdf,
        "PCA: varianza y proyeccion",
        [FIG_DIR / "pca_varianza.png", FIG_DIR / "pca_2d.png"],
        captions=["Varianza acumulada", "PC1 vs PC2"],
        subtitle="La separacion mejora parcialmente, pero no resuelve por completo la confusiones entre clases.",
        cols=1,
    )
    add_image_page(
        pdf,
        "PCA en 3D",
        [FIG_DIR / "pca_3d.png"],
        captions=["PC1 vs PC2 vs PC3"],
        subtitle="La vista 3D ayuda a entender la estructura global del espacio de caracteristicas.",
        cols=1,
    )
    add_text_page(pdf, "9. Validacion cruzada e hiperparametros", cv_body, footer="Se uso StratifiedKFold y busqueda de hiperparametros en modelos clave")
    add_text_page(pdf, "10. Uso de Pipeline", pipeline_body, footer="Se compararon pipelines con y sin PCA")
    add_text_page(pdf, "11. Evaluacion final", eval_body, footer="Se analizo accuracy, precision, recall, F1, confusion matrix y errores visuales")
    add_image_page(
        pdf,
        "Matriz de confusion y errores",
        [FIG_DIR / "matriz_confusion_final.png", FIG_DIR / "predicciones_correctas.png", FIG_DIR / "predicciones_incorrectas.png"],
        captions=["Matriz de confusion final", "Predicciones correctas", "Predicciones incorrectas"],
        subtitle="Las confusiones dominantes aparecen entre T-shirt/top y Shirt.",
        cols=1,
    )
    add_image_page(
        pdf,
        "Comparacion completa de matrices",
        [FIG_DIR / "matrices_confusion_todos.png"],
        captions=["Todos los modelos"],
        subtitle="Permite comparar el comportamiento por clase de cada clasificador.",
        cols=1,
    )
    add_text_page(pdf, "12. Opcion de rechazo", reject_body, footer="Se analizo el trade-off entre exactitud aceptada y tasa de rechazo")
    add_image_page(
        pdf,
        "Analisis de opcion de rechazo",
        [FIG_DIR / "reject_option_analysis.png", FIG_DIR / "muestras_rechazadas.png"],
        captions=["Accuracy vs rechazo", "Muestras rechazadas"],
        subtitle="El rechazo mejora la confianza del sistema al costo de abstenerse en una fraccion de muestras.",
        cols=1,
    )
    add_text_page(pdf, "13. Conclusiones", conclusion_body, footer="El proyecto cumple la estructura tecnica solicitada por el enunciado")

print(f"PDF generado en: {PDF_PATH}")
