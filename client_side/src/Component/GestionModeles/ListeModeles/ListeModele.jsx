import React, { useState, useEffect, useCallback } from 'react'
import MUIDataTable from "mui-datatables";
import { Button, Card, Typography } from '@mui/material';
import Checkbox from '@mui/material/Checkbox';
import Container from '@mui/material/Container';
import CircularProgress from '@mui/material/CircularProgress';
import Grid from "@mui/material/Grid";
import Link from '@mui/material/Link';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogTitle from '@mui/material/DialogTitle';
import AjouterModele from '../AjouterModele/AjouterModele';
import ModifierModele from '../ModifierModele/ModifierModele';
import axios from 'axios'
import { red } from '@mui/material/colors';

const ListeModele = () => {

const [modelName, setModelName] = useState("")
const [rowIndex, setRowIndex] = useState(0)
const [listModeles, setListeModeles] = useState(null)
const [redListeModeles, setRedListeModeles] = useState([])
const [modele, setModele] = useState(null);
const [modeles, setModeles] = useState([]);
const [loading, setLoading] = useState(true);
const [checked, setChecked] = useState(true);
const [responsive, setResponsive] = useState("vertical");
const [tableBodyHeight, setTableBodyHeight] = useState("400px");
const [tableBodyMaxHeight, setTableBodyMaxHeight] = useState("");
const [openAjout, setOpenAjout] = useState(false);
const [openModif, setOpenModif] = useState(false);
const [modelNames, setModelNames] = useState([])

const handleCheck = (event) => {
    setChecked(event.target.checked);
};

// const data = [
//     ["SentAvg", true],
//     ["ParSeq", false]
// ];

function selectProps(...props){
    return function(obj){
      const newObj = {};
      props.forEach(name =>{
        newObj[name] = obj[name];
      });
      return newObj;
    }
  }

//   Charger la liste des modèles
  useEffect(() => {
    axios.get('http://localhost:8080/models').then( res => {
    const {data} = res;
    console.log(data)
    // console.log(modeles.data)
    if(data) {
        setListeModeles(data);
            console.log(listModeles)
            const newList = data.map(selectProps("name", "visibility"));
            // const liste_modeles = listModeles.map( obj => Object.values(obj) )
            // console.log(liste_modeles)
            var temp = newList.map( Object.values );
            setRedListeModeles(temp)
            // getModelNames()
            setLoading(false)
            // setModeles(liste_modeles)
    }
    })
  }, []);

const columns = [
    {
        name: "name",
        label: "Nom du modèle",
    },
    {
        name: "visibility",
        label: "Visibilité",
        options: {
        customBodyRender: () => {
            return (
            // <EtatVehiculeCol
            //     value={value}
            //     index={tableMeta.columnIndex}
            //     change={event => updateValue(event)}
            // />
            <Checkbox
                checked={checked}
                onChange={handleCheck}
                inputProps={{ 'aria-label': 'controlled' }}
            />
            )
        }
        }
    },
    {
        options: {
        viewColumns: false,
        filter: false,
        customBodyRenderLite: (dataIndex) => {
            return (
                <Button
                    href="#"
                    // onClick={(e) => { e.preventDefault(); onRowSelect(dataIndex); handleOpenModif()}}
                    onClick={(e) => { e.preventDefault(); onRowSelection(dataIndex); handleOpenModif()}}
                    style={{fontFamily:'Poppins', backgroundColor:'#5885FB', color: "white", textTransform: "capitalize"}}
                    variant="filled"
                >
                    Modifier
                </Button>
            );
        }
        }
    }
    ];

const onRowSelect = (dataIndex) => {
    console.log(dataIndex)
}

const onRowSelection = async(dataIndex) => {
    // onRowSelect(dataIndex)
    // setModele(dataIndex);
    setModelName(dataIndex[0]);
    let model = listModeles.find(modele => modele.name == modelName)
    // console.log(model)
    setModele(model);
    // if (!!props.setSel) props.setSel(rowData[0])
}

const getModelNames = () => {
    if(listModeles) {
        let extractedNames = listModeles.map(item => item["name"]);
        setModelNames(extractedNames)
        
    }
    console.log(modelNames)

}

const options = {
    filter: false,
    search: false,
    download:false,
    print:false,
    viewColumns:false,
    filterType: "dropdown",
    elevation:0,
    responsive,
    tableBodyHeight,
    tableBodyMaxHeight,
    searchPlaceholder: 'Saisir un nom ou un ID..',
    onRowClick: onRowSelection,
    textLabels: {
        // body: {
        // noMatch: loading ?
        // <CircularProgress /> :
        // 'Aucune donnée trouvée',
        // toolTip: "Trier",
        // },
        pagination: {
            next: "Page suivante",
            previous: "Page précédente",
            rowsPerPage: "Modèle par page:",
            displayRows: "/",
          },
        selectedRows: {
        text: "ligne(s) sélectionné(s)",
        delete: "Supprimer",
        deleteAria: "Supprimer les lignes choisies",
        },
    }
};

const [anchorEl, setAnchorEl] = React.useState(null);

const handleClick = (event) => {
    // setModele(modele);
    // console.log(modele);
    setAnchorEl(event.currentTarget);
};

const handleClose = () => {
    setAnchorEl(null);
  };

const handleOpenAjout = () => {
    setOpenAjout(true);
  };

const handleCloseAjout = () => {
    setOpenAjout(false);
};

const handleOpenModif = () => {
    setOpenModif(true);
    handleClose()
};

const handleCloseModif = () => {
    setOpenModif(false);
};
   
  return (
    <>
        <div className="main-content">
            <Grid container component="main" className='root'
                spacing={0}
                direction="row"
                alignItems="center"
                justifyContent="center"
                p={2}
                elevation={1}
                sx={{
                    backgroundColor:'#0035BD',
                    width: '400px',
                    margin: '30px 37% 20px',
                    border: '2px solid #0035BD',
                    borderRadius: '10px',
                    
                }}
            >       
                <Typography variant='h5' sx={{ fontFamily: 'Poppins', fontWeight: 700, color: 'white'}}>
                    Liste des modèles
                </Typography>
              </Grid>
            
            <Container fluid sx={{width:"800px", border: "0.5px solid black", borderRadius:"10px", boxShadow: '5px 5px 11px -8px rgba(0,0,0,0.50)'}}>
                <div style={{padding:"12px 12px 20px 12px", marginTop: '10px'}}>
                    <Button variant="contained" onClick={handleOpenAjout} style={{backgroundColor:"#EFBB00", textTransform:"capitalize", color:"white", fontWeight:'bold', width:'150px'}}>
                    + Ajouter
                    </Button>
                </div>
                {
                    loading == true ? 
                    <div style={{position: "relative", margin: "0px 50%"}}>
                    <CircularProgress/> 
                    </div>
                    : 
                    <MUIDataTable
                    data={redListeModeles}
                    columns={columns}
                    options={options}
                    /> 
                }
                    
                
            </Container>

        <Dialog onClose={handleCloseAjout} aria-labelledby="customized-dialog-title" open={openAjout} fullWidth='true' maxWidth='sm'>
            <AjouterModele handleCloseAjout={handleCloseAjout}/>
        </Dialog>
        <Dialog onClose={handleCloseModif} aria-labelledby="customized-dialog-title" open={openModif} fullWidth='true' maxWidth='sm'>
            <ModifierModele
            handleCloseModif={handleCloseModif} 
            modele={modele}
            />
        </Dialog>
      </div>
    </>
  )
}

export default ListeModele