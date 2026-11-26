# Guion de la presentación · Clasificación de polaridad en evaluaciones docentes

Duración total: unos 15 minutos. Cinco integrantes, tres diapositivas cada uno y unos 3 minutos por persona (unas 320 palabras). La portada la abre Nicolás.

Reglas para que todos hablen lo mismo:
- Cada bloque tiene una entrada y una salida que da paso al siguiente integrante.
- Las cifras clave aparecen una sola vez en todo el guion para no repetir.
- Lo que está entre corchetes es una acción, no se lee.

---

## Integrante 1 · Nicolás Gómez Giraldo · Problema, datos y desarrollo realizado (diapositivas 1 a 4)

**Diapositiva 1 · Portada**

Buenos días. Somos Nicolás, Samuel, Esteban, José y Jhon, y vamos a presentar nuestro proyecto de Sistemas Inteligentes: un clasificador de la polaridad de los comentarios que los estudiantes escriben al evaluar a sus profesores. Cada uno de nosotros va a presentar una parte.

**Diapositiva 2 · El problema**

Al final de cada semestre los estudiantes califican a sus profesores con escalas numéricas y, muchas veces, con un comentario libre. Las escalas son fáciles de promediar, pero la literatura muestra que dicen poco sobre lo que realmente aprenden los estudiantes. Los comentarios, en cambio, explican por qué un curso funcionó o no. El problema es el volumen: leer cientos de comentarios por curso no escala, y esa información termina sin usarse.

Nuestro objetivo fue clasificar automáticamente cada comentario en positivo, neutral o negativo, comparar distintas formas de representar el texto y, sobre todo, llevar el mejor modelo a una herramienta que un docente o una coordinación pueda usar.

**Diapositiva 3 · Los datos**

Partimos de tres archivos CSV entregados por el profesor, con comentarios de siete cursos de computación entre 2023 y 2024. Antes de entrenar hubo que limpiarlos: los archivos repetían la fila de encabezado, la etiqueta de polaridad venía escrita en nueve formatos distintos y había errores en los nombres de los cursos. Después de la depuración quedaron 617 comentarios: 42 % neutrales, 33 % negativos y 25 % positivos, como se ve en la gráfica.

Un hallazgo importante es que el 83 % de los comentarios mezcla elogios y críticas, con palabras como «pero», «aunque» o «sin embargo». Eso, como veremos, es lo que más dificulta la tarea.

**Diapositiva 4 · Desarrollo realizado**

El proyecto tuvo dos fases. En la primera seguimos siete pasos: exploración de los datos, preprocesamiento, división sin fuga de información, representación del texto, línea base, ajuste de hiperparámetros y evaluación final. Esa fase terminó en el paper.

En la segunda construimos el MVP: un núcleo de inferencia, una API, una interfaz web, el versionado del modelo, pruebas automáticas y un despliegue con Docker.

[Salida] Ahora Samuel les va a contar cómo entrenamos y elegimos el modelo.

---

## Integrante 2 · Samuel Rueda Sánchez · Modelo clasificador y métricas (diapositivas 5 a 7)

**Diapositiva 5 · Cómo se entrenó el modelo**

Gracias. Para que la evaluación fuera honesta, separamos el 80 % de los comentarios para entrenar y el 20 % para la prueba final: 493 y 124 comentarios. La división respeta la proporción de cada clase y agrupa los comentarios casi duplicados, para que un mismo texto no quede a ambos lados.

Comparamos cuatro formas de representar el texto: dos clásicas, TF-IDF de palabras y de caracteres, y dos semánticas, los modelos de embeddings MiniLM y MPNet, que convierten cada comentario en un vector que captura su significado. Sobre cada una entrenamos cuatro clasificadores: regresión logística, SVM lineal, Naive Bayes y Random Forest. Como referencia mínima usamos un clasificador que siempre responde la clase más frecuente. Los hiperparámetros se ajustaron con GridSearchCV y validación cruzada de cinco pliegues, eligiendo siempre por macro-F1, que da el mismo peso a las tres clases.

**Diapositiva 6 · La representación importa más que el clasificador**

Este es el resultado central del paper. Con embeddings, cualquier clasificador queda entre 0,67 y 0,72 de macro-F1; con TF-IDF, el mejor apenas llega a 0,63. Es decir, cerca de nueve puntos de diferencia. En cambio, entre los cuatro clasificadores sobre embeddings las diferencias son pequeñas. La conclusión es que importa más cómo se representa el texto que el algoritmo que se usa después.

**Diapositiva 7 · Modelo final**

El mejor en validación cruzada fue MPNet con Random Forest, y lo evaluamos una sola vez sobre los 124 comentarios de prueba: macro-F1 de 0,681 y exactitud de 0,694. Es más del triple que la línea base, que obtiene 0,197.

La matriz de confusión muestra que el modelo casi nunca confunde los extremos: ningún negativo se predijo como positivo. Su debilidad es que tiende a refugiarse en la clase neutral: de 31 positivos, 15 los clasificó como neutrales. Al revisar los errores encontramos que dos de cada tres son comentarios mixtos, y una auditoría de las etiquetas mostró que el 8 % son incoherentes. El límite actual está más en los datos que en el modelo.

[Salida] Con el modelo elegido, Esteban les explica cómo lo integramos en la herramienta.

---

## Integrante 3 · Esteban Muñoz Serrano · Arquitectura, tecnologías e integración (diapositivas 8 a 10)

**Diapositiva 8 · Arquitectura**

Gracias. El diagrama muestra el recorrido completo de izquierda a derecha. Primero están los datos y el script que los depura. Luego el entrenamiento, que se hace una sola vez y produce tres archivos: el modelo guardado, sus metadatos y un registro de versiones. En el centro está el núcleo de inferencia, un módulo llamado predictor, que es la pieza clave: carga el modelo, lo valida y produce las predicciones. A la derecha, dos interfaces reutilizan ese mismo núcleo: una API para que otros sistemas consulten el modelo y una interfaz web para el usuario final. Como ambas comparten el núcleo, una predicción siempre da lo mismo sin importar por dónde se pida.

**Diapositiva 9 · Tecnologías**

Todo está escrito en Python. Para los datos y el modelo usamos pandas, scikit-learn y sentence-transformers sobre PyTorch en CPU; el modelo se guarda con joblib. El servicio se construyó con FastAPI, Uvicorn y Pydantic, que valida automáticamente las peticiones. La interfaz está hecha en Streamlit. Para la calidad usamos pytest y, para el despliegue, Docker y Docker Compose. Elegimos FastAPI porque genera la documentación de la API de forma automática, y Streamlit porque permite construir una interfaz completa solo con Python, sin escribir HTML ni JavaScript.

**Diapositiva 10 · Integración del modelo**

El modelo completo, el codificador de embeddings y el Random Forest, se guarda en un solo archivo y se registra con su huella digital SHA-256. Cuando la aplicación arranca, comprueba que esa huella coincida: si alguien cambia el archivo, el modelo no se carga. También precarga el codificador para que la primera petición no tarde; después, cada comentario se clasifica en unos 16 milisegundos.

La salida es estructurada: además de la polaridad, entrega la probabilidad de cada clase, la confianza, el margen frente a la segunda clase y una marca que indica si conviene revisión humana. La captura muestra la documentación interactiva de la API, que se genera sola, con sus endpoints para un comentario, para un lote y para un archivo CSV.

[Salida] José les muestra cómo instalarla y ponerla a funcionar.

---

## Integrante 4 · José Harvey Barrera Ramos · Instalación, ejecución y elementos destacados (diapositivas 11 a 13)

**Diapositiva 11 · Instalación**

Gracias. Los requisitos son pocos: Python 3.12 o superior, espacio para PyTorch y el modelo de embeddings, y conexión a internet solo la primera vez, para descargar MPNet. Después todo funciona sin conexión. Hay dos formas de instalar la herramienta. La primera es con Python: se crea un entorno virtual, se instala PyTorch para CPU y luego las dependencias del archivo requirements-mvp. La segunda no necesita Python: con Docker basta un solo comando, docker compose up, que construye la imagen y levanta la API y la interfaz al mismo tiempo. En cualquiera de los dos casos, un script de validación confirma que el modelo carga bien, que clasifica correctamente tres casos de ejemplo y mide la latencia.

**Diapositiva 12 · Ejecución y uso**

Para usarla se ejecuta un comando por componente. La interfaz se abre con streamlit run y queda disponible en el navegador, en el puerto 8501. La API se arranca con uvicorn en el puerto 8000, y su documentación está en la ruta /docs. La interfaz funciona sola, sin la API; la API sirve cuando otro sistema necesita consultar el modelo. Esta es la pantalla de inicio: a la izquierda, el modelo en uso y sus métricas; al centro, las pestañas para un comentario, para un CSV y la ficha del modelo.

**Diapositiva 13 · Elementos destacados**

Queremos destacar seis decisiones. Primero, la revisión humana: si la confianza es menor que 50 % o si las dos clases más probables están muy cerca, el sistema lo advierte en lugar de dar una respuesta falsa de seguridad. Segundo, el versionado: se puede volver a una versión anterior del modelo sin tocar el código. Tercero, la validación de entradas: textos vacíos, demasiado cortos o demasiado largos se señalan. Cuarto, la privacidad: todo se procesa localmente y los registros no guardan comentarios. Quinto, la calidad: 28 pruebas automáticas, cuatro de ellas con el modelo real. Y sexto, la interfaz admite tema claro y oscuro.

[Salida] Para terminar, Jhon les muestra la herramienta en funcionamiento.

---

## Integrante 5 · Jhon David Rios Aguirre · Funcionamiento y conclusiones (diapositivas 14 a 16)

**Diapositiva 14 · Clasificar un comentario**

Gracias. [Si hay demo en vivo, cambiar a la aplicación; si no, usar las capturas.] En la pestaña de comentario individual se escribe o se pega un comentario y se pulsa Clasificar. En el primer ejemplo, «El profesor explica con mucha claridad, los ejemplos fueron excelentes y aprendí muchísimo», el sistema responde positivo con 68 % de confianza y no pide revisión.

El segundo ejemplo es más interesante: «El profesor explica muy bien, pero las evaluaciones son injustas». Es un comentario mixto. El modelo se inclina por negativo, pero con solo 43 % de confianza y casi empatado con neutral, así que lo marca para revisión humana. Esa es justamente la idea: el sistema ayuda a priorizar y la persona decide en los casos dudosos.

**Diapositiva 15 · Clasificar un archivo CSV**

Para muchos comentarios se usa la pestaña de carga por CSV. Se sube el archivo, la aplicación detecta sola la columna de comentarios y muestra una vista previa. Si el archivo trae una columna de identificador o de aspecto, también la aprovecha. Al pulsar Clasificar archivo aparecen los totales por clase, una gráfica y una tabla con la polaridad, la confianza y la marca de revisión de cada comentario. Un interruptor permite ver solo los casos que requieren revisión, y los resultados se descargan como CSV para trabajarlos en Excel. Con los 617 comentarios del proyecto, el proceso tarda unos 12 segundos.

**Diapositiva 16 · Conclusiones**

Para cerrar, tres conclusiones. Primera: los embeddings semánticos superan a TF-IDF en cerca de nueve puntos de macro-F1. Segunda: la calidad de las etiquetas limita el desempeño más que la elección del clasificador. Tercera: el modelo ya no se queda en un notebook; está integrado en una herramienta verificable, probada y fácil de instalar.

Como trabajo futuro proponemos reetiquetar los datos con criterios claros, pasar de una polaridad global a una polaridad por aspecto, probar el ajuste fino de un transformer en español y agregar un filtro para textos que no son evaluaciones.

Muchas gracias. Quedamos atentos a sus preguntas.

---

## Guía para la demo en vivo (Jhon)

Preparación, antes de la exposición:
1. Abrir una terminal en la raíz del repositorio y ejecutar `venv\Scripts\python -m streamlit run proyecto/app/streamlit_app.py`.
2. Abrir `http://localhost:8501` y esperar a que cargue el modelo (unos 7 segundos la primera vez).
3. Tener a mano un CSV corto (20 a 30 comentarios) con una columna `feedback` o `comentario`.
4. Dejar abiertas las capturas de la presentación por si la demo falla.

Durante la demo:
1. Pegar el comentario positivo y pulsar Clasificar.
2. Pegar el comentario mixto y mostrar la alerta de revisión humana.
3. Ir a Carga por CSV, subir el archivo, pulsar Clasificar archivo y activar «Mostrar solo casos que requieren revisión».
4. Pulsar Descargar predicciones.

Ejemplos que funcionan bien:
- Positivo: «El profesor explica con mucha claridad, los ejemplos fueron excelentes y aprendí muchísimo en el curso.»
- Mixto, marcado para revisión: «El profesor explica muy bien, pero las evaluaciones son injustas»
- Negativo: «Las clases fueron desorganizadas y no hubo retroalimentación suficiente sobre los trabajos.»

Evitar elogios moderados sin adjetivos fuertes, como «domina el tema» o «clases dinámicas»: el modelo tiende a clasificarlos como neutrales.

## Posibles preguntas del profesor

- **¿Por qué Random Forest si las diferencias con los demás clasificadores no son significativas?** Se eligió por el mejor macro-F1 en validación cruzada, sin mirar el conjunto de prueba; el intervalo de confianza muestra que los modelos con embeddings son equivalentes.
- **¿Por qué macro-F1 y no exactitud?** Porque las clases están desbalanceadas y macro-F1 da el mismo peso a la clase positiva, que es la minoritaria.
- **¿Cómo evitaron la fuga de información?** División agrupada por casi-duplicados, vectorizadores ajustados solo con entrenamiento dentro del Pipeline y conjunto de prueba usado una sola vez.
- **¿El campo aspecto lo predice el modelo?** No. El modelo predice la polaridad global; el campo aspecto queda preparado para una extensión futura.
- **¿Se envían los comentarios a algún servicio externo?** No. Los embeddings se calculan localmente y los registros no guardan textos.
