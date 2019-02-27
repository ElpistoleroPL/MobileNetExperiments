# MobileNetExperiments

## Opis plików:
* *mobilenet_imagenet_experiment.py* -> odpowiada za trening sieci <br/>
* *utils.py* -> helper do ładowania optymizera i modelu sieci <br/>
* *model_mobilenet.py* -> architektura MobileNet w różnych konfiguracjach (tu raczej nie będzie żadnych zmian podczas treningu, jedynie gdy chcielibyśmy inne warianty mobilenetu przetestować) <br/>

<br/>

### Uruchamianie eksprymentu
*mobilenet_imagenet_experiment.py* przyjmuje następujące argumenty: <br/>
*  --n NAME nazwa eksperymentu, statytyki z poszczególnych epok zostaną zapisane w pliku którego nazwa będzi kończyć się nazwą eksperymentu. <br/>
* [--b BATCH_SIZE] rozmiar batha, domyślnie 32 <br/>
* [--l LR] learning_rate, domyślnie: wartośc zdefiniowana w pliku *utils.py* dla danego optymalizatora <br/>
* [--c CONFIG] id konfiguracji zdefiniowanej w pliku *utils.py* najlepiej w pracy magisterskiej spisywała się 3, 5,6,7 zostały dodane na etapie testowania na ImageNet <br/>
* [--e EPOCHS] liczba epok, domyślnie 15 <br/>
* [--m MODEL] id modelu,w naszym przypadku model 0 to oryginalny MobileNet i model 8 to wersja modelu który najlepiej na magisterce się spisywał <br/>
* [--a NET_PARAM] parametr alpa, domyślnie 1.0, dla zredukowanego modelu 8 powinien wynosić 0.92 <br/>
<br/>
Zazwyczaj uruchamiam kilka eksperymentów jeden po drugim, przykładowe konfiguracje w pliku *runImG0.sh* <br/>


