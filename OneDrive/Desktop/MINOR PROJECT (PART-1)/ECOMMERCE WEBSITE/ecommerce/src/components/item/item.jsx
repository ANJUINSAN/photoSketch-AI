import React, { useState } from "react";
import './item.css'
import { Link } from "react-router-dom";
import heart from '../images/Assets/heart.svg'
import likeheart from '../images/Assets/love.png'

export default function Item(props) {
  const [liked,setliked]=useState(true);

 
  return (
    <div className="item">
     <Link to={`/product/${props.id}`}>
     <img className="scrollimg" onClick={window.scrollTo(0,0)} src={props.image}>
     </img>
    
     </Link> 
     <div className="item-cont">
      <p>{props.name}
      </p>
      
      <button className="likebtn" onClick={()=>{setliked(!liked)}}>
        {liked?<img src={heart} alt="" />:<img src={likeheart} alt="" />}</button>
     </div>
      <div className="item-prices">
        <div className="item-price-new">{props.new_price} $</div>
      </div>
      <div className="item-price-old">
        <div>{props.old_price} $</div>
      </div>
    </div>
  );
}
