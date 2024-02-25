
function getNextElment() {
    if (is_explore_page)
        var url = "/api/saved";
    else
        var url = "/api/data";
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open("GET", url, false);
    xmlHttp.send(null);
    return JSON.parse(xmlHttp.responseText);
}

function getContributionsBy(name) {
    $.ajax({
        type: 'POST',
        url: "/api/getCon",
        data: {'Reviewed by': name}, //How can I preview this?
        dataType: 'json',
        success: function(d){
            document.getElementById('num_cont').value = d.num_cont;
        }
      });
}

function getContributionsNames() {
    var url = "/api/getConNames";
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open("GET", url, false);
    xmlHttp.send(null);
    return JSON.parse(xmlHttp.responseText);
}

function setUpBarGraph(){
    // Initialize the echarts instance based on the prepared dom
    var myChart = echarts.init(document.getElementById('main'));
    var contributers = getContributionsNames()
    // Specify the configuration items and data for the chart
    var source = []
    for (var key in contributers){
        source.push([key, contributers[key]])
      }
    console.log(source)
    var option = {
        dataset: [
            {
             dimensions: ['name', 'contributers'],
              source: source
            },
            {
                transform: [
                    {
                      type: 'sort',
                      config: { dimension: 'contributers', order: 'desc' }
                    }
                  ]
                }
          ],
      title: {
        text: ''
      },
      tooltip: {},
      legend: {
        data: ['Contributers']
      },
      xAxis: {
        type: 'category'
      },
      yAxis: {},
      series: [
        {
          name: 'Contributers',
          type: 'bar',
          itemStyle: {
            // HERE IS THE IMPORTANT PART
            color: "rgba(13,202,240,1)"
          },
          datasetIndex: 1
        }
      ]
    };

    myChart.setOption(option);
}

function getNext() {
  element = getNextElment();
  document.getElementById('text').value = element['text'];
  document.getElementById('subset').innerHTML = 'Dataset Name: ' + element['dataset_name'];
  document.getElementById('HiddenDatasetName').value =  element['dataset_name'];


  if (is_explore_page) {
      document.getElementById('num_rem').innerHTML = 'Remaining: ' + element['num_rem'];
      document.getElementById('num_contr').innerHTML = 'Total Contributions: ' + element['num_contr'];
  } else {
      document.getElementById('num_rem').innerHTML = 'Remaining: ' + element['num_rem'];
      document.getElementById('num_contr').innerHTML = 'Total Contributions: ' + element['num_contr'];
  }

  document.getElementById('index_input').value = element['index'];
  document.getElementById('index_text').innerHTML = 'Text#: ' + element['index'];
  document.getElementById('index_file').innerHTML = 'File#: ' + element['index_file'];
  document.getElementById('Hiddenindexfile').value = element['index_file'];
  document.getElementById('index_line').innerHTML = 'json line#: ' + element['index_line'];
  document.getElementById('Hiddenindexline').value = element['index_line'];
  document.getElementById('Reviewed by').value = curr_reviewer;
  console.log("Current Reviewer", curr_reviewer);

  if (is_explore_page) {
      setUpBarGraph();
  }
}

$(".edittable").on('change', function () {
    changed = true
});

$(".changed").on('change', function () {
    curr_reviewer = this.value
});

function checkChanges() {
    if (!changed) {
        $('#exampleModal').modal('show');
    } else {
        submitForm()
    }
}

function submitForm() {
    document.getElementById('theForm').submit();
    num_cont += 1
    document.getElementById('num_rem').value =  num_cont;
    document.getElementById('num_contr').value =  num_cont;


}

is_explore_page = window.location.pathname == '/explore'
if (is_explore_page) {
    $('#btnSubmit').hide();
}

$(document).on('submit','#theForm',function(e)
    {
      e.preventDefault();
      $.ajax({
        type:'POST',
        url:'/api/submit',
        data:$('form').serialize(),
        success:function()
        {
            getNext();
            getContributionsBy(curr_reviewer)

        }
      })
    });

num_cont = 0
curr_reviewer = ""
changed = true
getNext()
