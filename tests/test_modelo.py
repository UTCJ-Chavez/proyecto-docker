from app.modelo import cargar_datos, entrenar, calcular_metricas


def test_cargar_datos_divide_correctamente():
    X_train, X_test, y_train, y_test = cargar_datos(test_size=0.2)
    assert len(X_train) == 120
    assert len(X_test) == 30


def test_entrenar_devuelve_modelo_con_predict():
    X_train, _, y_train, _ = cargar_datos()
    modelo = entrenar(X_train, y_train)
    assert hasattr(modelo, "predict")


def test_metricas_perfectas():
    m = calcular_metricas([0, 1, 2], [0, 1, 2])
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_accuracy_minimo_aceptable():
    X_train, X_test, y_train, y_test = cargar_datos()
    modelo = entrenar(X_train, y_train)
    m = calcular_metricas(y_test, modelo.predict(X_test))
    assert m["accuracy"] >= 0.85


def test_main_genera_archivo_metricas(tmp_path, monkeypatch):
    import json
    from app.modelo import main
    monkeypatch.chdir(tmp_path)
    resultado = main()
    datos = json.loads((tmp_path / "metrics.json").read_text())
    assert datos == resultado
