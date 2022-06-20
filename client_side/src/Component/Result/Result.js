import React from 'react'
import Card from '@mui/material/Card';
import Box from '@mui/material/Box';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import Typography from '@mui/material/Typography';
import Brightness1RoundedIcon from '@mui/icons-material/Brightness1Rounded';
import Button from '@mui/material/Button';
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './Result.css';
import { styled } from '@mui/material/styles';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell, { tableCellClasses } from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Paper from '@mui/material/Paper';
import AddCircleOutlineIcon from '@mui/icons-material/AddCircleOutline';
import RemoveCircleOutlineIcon from '@mui/icons-material/RemoveCircleOutline';
import Looks3Icon from '@mui/icons-material/Looks3';

const Result = ({ hidden, scoreResult, isEmpty, chartData, chart, chartLength, table }) => {
  const [displayTable , setDisplay] = React.useState(false);
  const [buttonTexte , setButtonTexte] = React.useState("Afficher Plus");
  const [icon, setIcon] = React.useState(<AddCircleOutlineIcon/>)
  const handleDisplayTable = (e) => {
    e.preventDefault();
    if(!displayTable){
      setDisplay(true)
      setButtonTexte("Réduire le tableau")
      setIcon(<RemoveCircleOutlineIcon/>)
    }
    else{
      setDisplay(false)
      setButtonTexte("Afficher Plus")
      setIcon(<AddCircleOutlineIcon/>)
    }
    
  };

  function Item(props) {
    const { sx, ...other } = props;
    return (
      <Box
        sx={{
          p: 1,
          bgcolor: 'transparent',
          color: (theme) => (theme.palette.mode === 'dark' ? 'grey.300' : 'grey.800'),
          fontSize: '0.875rem',
          ...sx,
        }}
        {...other}
      />
    );
  }

  function Score({ scoreResult }) {
    switch (scoreResult) {
      case null:
        return <></>
      default:
        return <BasicCard score={scoreResult} />
    }
  }

  function BasicCard({ score }) {
    let val
    if (score > 2) {
      val = <Typography variant="h6" component="div" color="#079615">
        Score de cohérence : {score}
      </Typography>
    }
    else if (score > 1 && score <= 2) {
      val = <Typography variant="h6" component="div" color="#FF9A02">
        Score de cohérence : {score}
      </Typography>
    }
    else if (score >= 0 && score <= 1) {
      val = <Typography variant="h6" component="div" color="#E33A3A">
        Score de cohérence : {score}
      </Typography>
    }
    else {
      val = <Typography variant="h6" component="div">
        Score de coherence : {score}
      </Typography>
    }
    return (
      <div className='result'>
        <Card sx={{ minWidth: 275, border: 1 }}>
          <CardContent>
            <Typography variant="h6" component="div">
              {val}
            </Typography>
          </CardContent>
        </Card>
      </div>
    );
  }

  function RenderChart({ isEmpty, chartData }) {
    if (isEmpty === true) {
      return <></>
    }
    else {
      return (
        <>

          <div style={{ display: 'block' }}>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <BarChart
                width={500}
                height={300}
                data={chartData}
                margin={{
                  top: 0,
                  right: 30,
                  left: 20,
                  bottom: 5,
                }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" />
                <YAxis tickCount={chartLength} />
                <Tooltip />
                <Legend />
                <Bar dataKey="score" fill="#ffab00" />
              </BarChart>
            </div>
            <div>
              <div style={{ width: '100%', display: 'flex', justifyContent: 'center', marginBottom: '50px' }} >
                <Typography variant='body2' sx={{ fontSize: 18, fontFamily: 'Didact Gothic' }} color="#000" gutterBottom>
                  Nombre de documents par classe de cohérence
                </Typography>
              </div>
            </div>
          </div>
          <Button variant="outlined"  endIcon={icon}  onClick={handleDisplayTable}>{buttonTexte}</Button>
          <RenderTable rows={table} displayTable = {displayTable}></RenderTable>

        </>

      )
    }
  }

  function Render({ chart }) {
    if (chart === true) {
      return (
        <>
          <div>
            <RenderChart isEmpty={isEmpty} chartData={chartData} />
          </div>

        </>
      )
    }
    else {
      return (
        <>
          <Score scoreResult={scoreResult} />
          <div className='cards-key'>
            <div className='card1'>
              <div><Brightness1RoundedIcon sx={{ color: "#079615" }} /></div>
              <p id='scoreCard'> 3 (élevé)</p>
            </div>
            <div className='card1'>
              <div><Brightness1RoundedIcon sx={{ color: "#FF9A02" }} /></div>
              <p id='scoreCard'> 2 (moyen)</p>
            </div>
            <div className='card1'>
              <div><Brightness1RoundedIcon sx={{ color: "#E33A3A" }} /></div>
              <p id='scoreCard'> 1 (bas)</p>
            </div>
          </div>
        </>
      )
    }
  }

  const StyledTableCell = styled(TableCell)(({ theme }) => ({
    [`&.${tableCellClasses.head}`]: {
      backgroundColor: "#0288d1",
      color: theme.palette.common.white,
    },
    [`&.${tableCellClasses.body}`]: {
      fontSize: 14,
    },
  }));

  const StyledTableRow = styled(TableRow)(({ theme }) => ({
    '&:nth-of-type(odd)': {
      backgroundColor: theme.palette.action.hover,
    },
    // hide last border
    '&:last-child td, &:last-child th': {
      border: 0,
    },
  }));

  function RenderTable({ rows, displayTable }) {
    if (displayTable ==false){
      return
    }else{
    return (
      <div>
        <Table sx={{ minWidth: 70, maxWidth: 1000 , height: "200px", overflow:"scroll"}} aria-label="customized table">
          <TableHead>
            <TableRow>
              <StyledTableCell sx={{fontFamily : 'Didact Gothic', fontWeight :"bold"}} >ID du document</StyledTableCell>
              <StyledTableCell align="center" sx={{fontFamily : 'Didact Gothic', fontWeight :"bold"}}>Texte</StyledTableCell>
              <StyledTableCell align="left" sx={{fontFamily : 'Didact Gothic', fontWeight :"bold"}}>Score original</StyledTableCell>
              <StyledTableCell align="left" sx={{fontFamily : 'Didact Gothic', fontWeight :"bold"}}>Score prédit</StyledTableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.map((row) => (
              <StyledTableRow key={row.text_id}>
                <StyledTableCell component="th" scope="row">
                  {row.text_id}
                </StyledTableCell>
                <StyledTableCell align="justify" sx={{width: "400px", height : "200px", fontFamily : 'Didact Gothic'}}><div style={{width: "400px", height: "200px" , overflow: "auto"}}>{row.text}</div></StyledTableCell>
                <StyledTableCell align="center" sx={{fontFamily : 'Didact Gothic'}}>{row.original_score}</StyledTableCell>
                <StyledTableCell align="center" sx={{ fontWeight: "bold" , fontFamily : 'Didact Gothic'}}>{row.predicted_score}</StyledTableCell>
              </StyledTableRow>
            ))}
          </TableBody>
        </Table>
      </div>

    );
            }

  }
  return (
    <>
      <Box
          sx={{
            display: 'flex',
            flexDirection: 'row',
            justifyContent: 'space-between',
            position: 'relative',
            width: '75%',
            height: '60px',
            margin: '10px 133px'
          }}
        >
          <Item sx={{ backgroundColor: 'none', height: '50px', width: '100%' }}>
            <Typography variant="h5" sx={{ fontFamily: 'Poppins', fontWeight: 500, color: '#5885FB' }}><Looks3Icon sx={{ margin: '0 18px', height: '4%', width: '4%', color: "#ffab00" }} />Visualisation des résultats</Typography>
          </Item>
        </Box>
      <div id='evalSection' hidden={hidden}>
        <Render chart={chart} />
      </div>
    </>
    
  )
}

export default Result